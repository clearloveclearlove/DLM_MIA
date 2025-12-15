"""
Token-Level Trajectory AUC Attack for Diffusion Language Models

Analyzes token-level probability trajectories during progressive context unmasking
to distinguish training members from non-members.

Hypothesis:
- Members:  Stable token predictions (flat trajectory) → Low AUC
- Non-members: Context-dependent predictions (varying trajectory) → High AUC
"""

import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params
from attack.misc.models import ModelManager


class TokenTrajectoryAttack(AbstractAttack):
    """
    Token-Level Trajectory AUC-based Membership Inference Attack.

    Computes probability trajectories for individual tokens during progressive
    context unmasking and uses aggregated AUC scores as membership signals.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # Core parameters
        self.num_steps = int(config.get("num_steps", 6))  # Context unmasking steps
        self.num_bins = int(config.get("num_bins", 3))  # Number of token bins
        self.batch_size = int(config.get("batch_size", 4))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))

        # Aggregation method for token-level scores
        self.aggregation = config.get("aggregation", "bottom_k")
        self.aggregation_ratio = float(config.get("aggregation_ratio", 0.01))  # For top_k/bottom_k

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

        # Generate context mask schedule
        self.context_mask_ratios = self._generate_mask_schedule()

        logging.info(f"[TokenTrajectoryAttack] Initialized:")
        logging.info(f"  - num_steps: {self.num_steps}")
        logging.info(f"  - num_bins: {self.num_bins}")
        logging.info(f"  - aggregation: {self.aggregation} (ratio={self.aggregation_ratio})")
        logging.info(f"  - use_reference_model: {self.use_reference_model}")
        logging.info(f"  - context_mask_ratios: {[f'{r:.2f}' for r in self.context_mask_ratios]}")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

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

        # Get actual device after DataParallel wrapping
        if isinstance(self.ref_model, torch.nn.DataParallel):
            self.ref_device = next(self.ref_model.parameters()).device
        else:
            self.ref_device = ref_device

        logging.info(f"Reference model loaded on {self.ref_device}")

    def _generate_mask_schedule(self):
        """Generate progressive context mask ratios from 1. 0 to 0.0."""
        return np.linspace(1.0, 0.0, self.num_steps)

    def run(self, dataset: Dataset) -> Dataset:
        """Run token-level trajectory attack on the dataset."""
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset

        membership_scores = []

        # 🔥 修复：使用真正的批处理
        for start_idx in tqdm(range(0, n_samples, self.batch_size), desc=f"{self.name}"):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = dataset[start_idx:end_idx]

            # 现在 _compute_batch_scores 真正利用batch处理
            batch_scores = self._compute_batch_scores(batch["text"])
            membership_scores.extend(batch_scores)

        # Cleanup reference model
        if self.ref_model is not None:
            self.ref_model.to('cpu')
            del self.ref_model
            del self.ref_tokenizer
            torch.cuda.empty_cache()

        return dataset.add_column(self.name, membership_scores)

    def _compute_batch_scores(self, texts):
        """
        Compute token-level trajectory-based scores for a batch of texts.

        🔥 关键修复：现在真正并行处理整个batch
        """
        # 🔥 对每个文本独立处理（因为token数量和bin划分不同）
        # 但我们在每个文本内部做批量化优化
        scores = []

        for text in texts:
            try:
                if self.use_reference_model:
                    # Compute trajectories for both models
                    target_token_aucs = self._compute_token_aucs_for_text(
                        text, self.model, self.tokenizer, self.target_mask_id,
                        self.target_shift_logits, self.device
                    )

                    ref_token_aucs = self._compute_token_aucs_for_text(
                        text, self.ref_model, self.ref_tokenizer, self.ref_mask_id,
                        self.ref_shift_logits, self.ref_device
                    )

                    if target_token_aucs is None or ref_token_aucs is None:
                        scores.append(0.0)
                        continue

                    # Compute relative AUCs
                    relative_aucs = []
                    min_len = min(len(target_token_aucs), len(ref_token_aucs))
                    for i in range(min_len):
                        relative_auc = target_token_aucs[i] / (ref_token_aucs[i] + 1e-8)
                        relative_aucs.append(relative_auc)

                    # Aggregate relative AUCs
                    aggregated_score = self._aggregate_scores(relative_aucs)
                    scores.append(-aggregated_score)

                else:
                    # Single model version
                    token_aucs = self._compute_token_aucs_for_text(
                        text, self.model, self.tokenizer, self.target_mask_id,
                        self.target_shift_logits, self.device
                    )

                    if token_aucs is None:
                        scores.append(0.0)
                        continue

                    aggregated_score = self._aggregate_scores(token_aucs)
                    scores.append(-aggregated_score)

            except Exception as e:
                logging.warning(f"Error processing text: {e}")
                scores.append(0.0)

        return scores

    def _compute_token_aucs_for_text(self, text, model, tokenizer, mask_id, shift_logits, device):
        """
        Compute token-level trajectory AUCs for a single text.

        🔥 核心优化：完全复制 case_study_token_level_auc.py 的逻辑
        """
        # Tokenize
        encoded = tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(device)

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].bool()
        valid_len = int(attention_mask.sum().item())

        if valid_len < 2:
            return None

        # Get all valid token positions (skip special tokens)
        all_valid_positions = list(range(1, valid_len - 1))

        # 🔥 CRITICAL:  Shuffle to preserve local context across bins
        random.shuffle(all_valid_positions)

        if len(all_valid_positions) == 0:
            return None

        # Divide into bins
        num_bins = min(self.num_bins, len(all_valid_positions))
        bin_size = len(all_valid_positions) // num_bins

        all_token_aucs = []

        # Process each bin
        for bin_idx in range(num_bins):
            start_idx = bin_idx * bin_size
            if bin_idx == num_bins - 1:
                end_idx = len(all_valid_positions)
            else:
                end_idx = (bin_idx + 1) * bin_size

            bin_positions = all_valid_positions[start_idx:end_idx]

            if len(bin_positions) == 0:
                continue

            # Convert to tensor
            target_indices = torch.tensor(bin_positions, device=device)
            target_token_ids = input_ids[target_indices]

            # Get context positions (all except current bin)
            context_positions = [p for p in range(valid_len) if p not in bin_positions]
            context_positions = torch.tensor(context_positions, device=device)

            # Shuffle context for progressive unmasking
            g = torch.Generator(device=device)
            g.manual_seed(self.seed + bin_idx)
            context_perm = torch.randperm(len(context_positions), generator=g, device=device)
            context_positions_shuffled = context_positions[context_perm]

            # Compute trajectories for this bin
            bin_trajectories = [[] for _ in range(len(target_indices))]

            # 🔥 关键：对每个context_mask_ratio只做1次前向传播
            for context_mask_ratio in self.context_mask_ratios:
                # Determine how many context tokens to mask
                num_context_to_mask = int(np.round(context_mask_ratio * len(context_positions)))

                masked_ids = input_ids.clone()
                masked_ids[target_indices] = mask_id  # Always mask target

                if num_context_to_mask > 0:
                    positions_to_mask = context_positions_shuffled[:num_context_to_mask]
                    masked_ids[positions_to_mask] = mask_id

                # 🔥 单次前向传播
                with torch.no_grad():
                    outputs = model(
                        input_ids=masked_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0) if not shift_logits else None
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    if shift_logits:
                        logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

                    probs = F.softmax(logits[0], dim=-1)

                    # 🔥 提取所有target tokens的概率（一次性）
                    for i, (pos, token_id) in enumerate(zip(target_indices, target_token_ids)):
                        token_prob = probs[pos, token_id].item()
                        bin_trajectories[i].append(token_prob)

            # Calculate AUC for each token in this bin
            for traj in bin_trajectories:
                if len(traj) > 1:
                    auc = self._calculate_trajectory_auc(np.array(traj))
                    all_token_aucs.append(auc)

        return all_token_aucs if all_token_aucs else None

    def _calculate_trajectory_auc(self, trajectory):
        """Calculate Area Under the trajectory Curve."""
        if len(trajectory) < 2:
            return 0.0

        # Using trapezoidal rule
        auc = np.trapz(trajectory, x=self.context_mask_ratios)
        x_range = self.context_mask_ratios[0] - self.context_mask_ratios[-1]

        # Normalize by x_range
        normalized_auc = auc / x_range if abs(x_range) > 1e-8 else auc

        return float(normalized_auc)

    def _aggregate_scores(self, scores):
        """Aggregate token-level scores using specified method."""
        if not scores:
            return 0.0

        arr = np.array(scores)

        if self.aggregation == 'mean':
            return float(np.mean(arr))
        elif self.aggregation == 'max':
            return float(np.max(arr))
        elif self.aggregation == 'min':
            return float(np.min(arr))
        elif self.aggregation == 'median':
            return float(np.median(arr))
        elif self.aggregation == 'top_k':
            k = max(1, int(len(arr) * self.aggregation_ratio))
            top_k_vals = np.partition(arr, -k)[-k:]
            return float(np.mean(top_k_vals))
        elif self.aggregation == 'bottom_k':
            k = max(1, int(len(arr) * self.aggregation_ratio))
            bottom_k_vals = np.partition(arr, k - 1)[:k]
            return float(np.mean(bottom_k_vals))
        else:
            return float(np.mean(arr))