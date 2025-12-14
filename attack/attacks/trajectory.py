"""
PPL Trajectory Attack for Diffusion Language Models

Analyzes the perplexity trajectory during progressive denoising (100% mask → 0% mask)
to distinguish training members from non-members.

Hypothesis:
- Members:  Fast PPL drop at high mask ratios → Small AUC
- Non-members: Slow PPL drop → Large AUC
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params
from attack.misc.models import ModelManager


class TrajectoryAttack(AbstractAttack):
    """
    PPL Trajectory-based Membership Inference Attack.

    Computes PPL at multiple progressive denoising steps and uses the
    Area Under the trajectory Curve (AUC) as membership score.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # Core parameters
        self.num_steps = int(config.get("num_steps", 11))  # Number of denoising steps
        self.num_runs = int(config.get("num_runs", 1))  # Monte Carlo sampling runs
        self.batch_size = int(config.get("batch_size", 4))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))

        # Use reference model for calibration
        self.use_reference_model = config.get("use_reference_model", False)
        self.ref_model = None
        self.ref_tokenizer = None

        # Model-specific parameters
        if 'model_mask_id' in config and 'model_shift_logits' in config:
            self.target_mask_id = config['model_mask_id']
            self.target_shift_logits = config['model_shift_logits']
        else:
            self.target_mask_id, self.target_shift_logits = get_model_nll_params(self.model)

        # Initialize reference model if needed
        if self.use_reference_model:
            self._init_reference_model(config)

        # Generate mask schedule
        self.mask_ratios = self._generate_mask_schedule()

        logging.info(f"[TrajectoryAttack] Initialized:")
        logging.info(f"  - num_steps: {self.num_steps}")
        logging.info(f"  - num_runs: {self.num_runs}")
        logging.info(f"  - use_reference_model: {self.use_reference_model}")
        logging.info(f"  - mask_ratios: {[f'{r:.2f}' for r in self.mask_ratios]}")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _init_reference_model(self, config):
        """Initialize reference model for calibration."""
        ref_model_path = config.get('reference_model_path')
        if not ref_model_path:
            raise ValueError("reference_model_path must be specified when use_reference_model=True")

        logging.info(f"Loading reference model from: {ref_model_path}")

        ref_device = torch.device(config.get("reference_device", "cuda"))
        self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
            ref_model_path, ref_model_path, ref_device
        )
        self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.ref_model)

        # 🔥 修复：获取 DataParallel 包装后的真实设备
        if isinstance(self.ref_model, torch.nn.DataParallel):
            self.ref_device = next(self.ref_model.parameters()).device
        else:
            self.ref_device = self.ref_model.device

        logging.info(f"Reference model loaded on {self.ref_device}")

    def _generate_mask_schedule(self):
        """Generate progressive mask ratios from 1. 0 to 0.0."""
        return np.linspace(1.0, 0.0, self.num_steps)

    def run(self, dataset: Dataset) -> Dataset:
        """Run trajectory attack on the dataset."""
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset

        membership_scores = []

        for start_idx in tqdm(range(0, n_samples, self.batch_size), desc=f"{self.name}"):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = dataset[start_idx:end_idx]
            batch_scores = self._compute_batch_scores(batch["text"])
            membership_scores.extend(batch_scores)

        # Cleanup
        if self.ref_model is not None:
            self.ref_model.to('cpu')
            del self.ref_model
            del self.ref_tokenizer
            torch.cuda.empty_cache()

        return dataset.add_column(self.name, membership_scores)

    def _compute_batch_scores(self, texts):
        """Compute trajectory-based scores for a batch of texts."""
        B = len(texts)

        # Tokenize
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        valid_lengths = attention_mask.sum(dim=1)

        # Compute trajectories
        if self.use_reference_model:
            # Calibrated version:  compute both target and reference trajectories
            target_trajectories = self._compute_trajectories(
                input_ids, attention_mask, valid_lengths,
                self.model, self.target_mask_id, self.target_shift_logits, self.device
            )

            # 🔥 修复：使用保存的 ref_device
            input_ids_ref = input_ids.clone().to(self.ref_device)
            attention_mask_ref = attention_mask.clone().to(self.ref_device)
            valid_lengths_ref = valid_lengths.clone().to(self.ref_device)

            ref_trajectories = self._compute_trajectories(
                input_ids_ref, attention_mask_ref, valid_lengths_ref,
                self.ref_model, self.ref_mask_id, self.ref_shift_logits, self.ref_device
            )

            # Compute relative AUC (target / reference)
            scores = []
            for i in range(B):
                target_auc = self._compute_auc(target_trajectories[i])
                ref_auc = self._compute_auc(ref_trajectories[i])
                # Lower relative AUC = member
                relative_auc = target_auc / (ref_auc + 1e-8)
                scores.append(-relative_auc)  # Negative so higher = member
        else:
            # Single model version
            trajectories = self._compute_trajectories(
                input_ids, attention_mask, valid_lengths,
                self.model, self.target_mask_id, self.target_shift_logits, self.device
            )

            # Compute AUC for each trajectory
            scores = []
            for i in range(B):
                auc = self._compute_auc(trajectories[i])
                # Lower AUC = faster drop = member
                scores.append(-auc)  # Negative so higher = member

        return scores

    def _compute_trajectories(self, input_ids, attention_mask, valid_lengths,
                              model, mask_id, shift_logits, device):
        """
        Compute PPL trajectories for a batch using progressive denoising.

        Returns:
            List[np.ndarray]: PPL trajectory for each sample, shape (num_steps,)
        """
        B = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Storage for trajectories
        trajectories = [[] for _ in range(B)]

        # Sample fixed mask positions once (for reproducibility across steps)
        # This ensures the same tokens are progressively revealed
        fixed_mask_positions = []
        for b in range(B):
            valid_len = int(valid_lengths[b].item())
            if valid_len > 0:
                # Get all valid positions
                valid_pos = torch.arange(valid_len, device=device)
                # Shuffle for randomness
                perm = torch.randperm(valid_len, device=device)
                fixed_mask_positions.append(valid_pos[perm])
            else:
                fixed_mask_positions.append(torch.tensor([], device=device))

        # Compute PPL at each denoising step
        for step_idx, mask_ratio in enumerate(self.mask_ratios):
            # Average over multiple runs
            step_ppls = [[] for _ in range(B)]

            for run_idx in range(self.num_runs):
                # Create masked input
                masked_ids = input_ids.clone()
                for b in range(B):
                    valid_len = int(valid_lengths[b].item())
                    if valid_len == 0:
                        continue

                    # Determine how many tokens to mask
                    num_to_mask = int(np.round(mask_ratio * valid_len))

                    if num_to_mask > 0:
                        # Use fixed positions from the beginning
                        positions_to_mask = fixed_mask_positions[b][: num_to_mask]
                        masked_ids[b, positions_to_mask] = mask_id

                # Forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=masked_ids,
                        attention_mask=attention_mask if not shift_logits else None
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    # Conditional logit shifting
                    if shift_logits:
                        logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

                    # Compute cross entropy loss (NLL)
                    ce = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        input_ids.view(-1),
                        reduction='none'
                    ).view(B, seq_len)

                    # Compute per-sample NLL (only on valid tokens)
                    for b in range(B):
                        valid_mask = attention_mask[b]
                        if valid_mask.sum() > 0:
                            sample_nll = ce[b, valid_mask].mean().item()
                            # Convert to PPL
                            # sample_ppl = np.exp(sample_nll)
                            step_ppls[b].append(sample_nll)

            # Average across runs and store
            for b in range(B):
                if step_ppls[b]:
                    avg_ppl = np.mean(step_ppls[b])
                    trajectories[b].append(avg_ppl)
                else:
                    # Fallback for empty sequences
                    trajectories[b].append(1.0)

        # Convert to numpy arrays
        return [np.array(traj) for traj in trajectories]

    def _compute_auc(self, trajectory):
        """
        Compute Area Under the trajectory Curve using trapezoidal rule.

        Args:
            trajectory: np.ndarray of shape (num_steps,) containing PPL values

        Returns:
            float:  AUC value (normalized by number of steps)
        """
        if len(trajectory) < 2:
            return 0.0

        # Use trapezoidal rule for integration
        # x-axis: mask_ratios (1.0 to 0.0)
        # y-axis: PPL values
        auc = np.trapz(trajectory, x=self.mask_ratios)

        # Normalize by x-axis range for comparability
        x_range = self.mask_ratios[0] - self.mask_ratios[-1]  # Should be 1.0
        normalized_auc = auc / x_range if x_range > 0 else auc

        return normalized_auc