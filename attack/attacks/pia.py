# https://arxiv.org/pdf/2305.18355
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
import warnings

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params


class PiaAttack(AbstractAttack):
    """
    PIA (Proximal Initialization Attack) adapted for diffusion language models.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # Core parameters
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))

        # PIA-specific parameters
        self.mask_ratio_t = float(config.get("mask_ratio_t", 0.3))
        self.norm_type = config.get("norm_type", "l2")  # "l1", "l2", or "cross_entropy"
        self.normalize_output = config.get("normalize_output", False)
        self.eps = float(config.get("eps", 1e-8))  # For numerical stability

        # Model-specific parameters
        if 'model_mask_id' in config and 'model_shift_logits' in config:
            self.mask_id = config['model_mask_id']
            self.shift_logits = config['model_shift_logits']
        else:
            self.mask_id, self.shift_logits = get_model_nll_params(self.model)

        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(
            f"PIA initialized: mask_ratio_t={self.mask_ratio_t}, norm_type={self.norm_type}, normalize={self.normalize_output}")

    def run(self, dataset: Dataset) -> Dataset:
        """Run PIA attack on the dataset."""
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset

        membership_scores = []

        for start_idx in tqdm(range(0, n_samples, self.batch_size), desc=f"{self.name}"):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = dataset[start_idx:end_idx]
            batch_scores = self._compute_batch_scores(batch["text"])
            membership_scores.extend(batch_scores)

        return dataset.add_column(self.name, membership_scores)

    def _compute_batch_scores(self, texts):
        """Compute PIA scores for a batch of texts."""
        batch_scores = []

        for text in texts:
            try:
                score = self._compute_pia_score(text)
                # Check for NaN/inf and replace with neutral score
                if not np.isfinite(score):
                    print(f"Warning: Non-finite PIA score {score}, replacing with 0.0")
                    score = 0.0
                batch_scores.append(float(score))
            except Exception as e:
                print(f"Error computing PIA score: {e}")
                batch_scores.append(0.0)

        return batch_scores

    def _compute_pia_score(self, text):
        """Compute PIA score with improved numerical stability."""
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()

        # Handle empty or very short sequences
        valid_positions = torch.where(attention_mask[0])[0]
        valid_len = len(valid_positions)

        if valid_len <= 1:
            return 0.0

        # Step 1: Get predictions at t=0 (no masking)
        with torch.no_grad():
            try:
                outputs_t0 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask if not self.shift_logits else None
                )
                logits_t0 = outputs_t0.logits if hasattr(outputs_t0, 'logits') else outputs_t0[0]

                if self.shift_logits:
                    logits_t0 = torch.cat([logits_t0[:, :1, :], logits_t0[:, :-1, :]], dim=1)

                # Check for NaN/inf in logits
                if not torch.isfinite(logits_t0).all():
                    print("Warning: Non-finite values in t=0 logits")
                    return 0.0

                # Use cross-entropy loss instead of probability distributions for stability
                ce_t0 = F.cross_entropy(
                    logits_t0.view(-1, logits_t0.size(-1)),
                    input_ids.view(-1),
                    reduction='none'
                ).view(input_ids.shape)

            except Exception as e:
                print(f"Error in t=0 forward pass: {e}")
                return 0.0

        # Step 2: Create masked version
        masked_input_ids = input_ids.clone()
        num_to_mask = max(1, int(self.mask_ratio_t * valid_len))
        num_to_mask = min(num_to_mask, valid_len - 1)  # Leave at least one token unmasked

        if num_to_mask == 0:
            return 0.0

        # Deterministic masking
        sample_hash = hash(text[:50]) % (2 ** 16)  # Use first 50 chars to avoid huge hashes
        rng = np.random.RandomState(sample_hash)

        try:
            mask_indices = rng.choice(valid_len, size=num_to_mask, replace=False)
        except ValueError:
            # Fallback if choice fails
            mask_indices = np.arange(min(num_to_mask, valid_len))

        positions_to_mask = valid_positions[mask_indices]
        masked_input_ids[0, positions_to_mask] = self.mask_id

        # Step 3: Get predictions on masked version
        with torch.no_grad():
            try:
                outputs_t = self.model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask if not self.shift_logits else None
                )
                logits_t = outputs_t.logits if hasattr(outputs_t, 'logits') else outputs_t[0]

                if self.shift_logits:
                    logits_t = torch.cat([logits_t[:, :1, :], logits_t[:, :-1, :]], dim=1)

                if not torch.isfinite(logits_t).all():
                    print("Warning: Non-finite values in masked logits")
                    return 0.0

                ce_t = F.cross_entropy(
                    logits_t.view(-1, logits_t.size(-1)),
                    input_ids.view(-1),
                    reduction='none'
                ).view(input_ids.shape)

            except Exception as e:
                print(f"Error in masked forward pass: {e}")
                return 0.0

        # Step 4: Compute error metric
        try:
            if self.norm_type == "cross_entropy":
                # Use difference in cross-entropy losses at masked positions
                ce_diff = ce_t[0, positions_to_mask] - ce_t0[0, positions_to_mask]
                error = torch.mean(ce_diff).item()

            elif self.norm_type == "l1":
                # L1 difference in logits at masked positions
                logit_diff = torch.abs(logits_t[0, positions_to_mask] - logits_t0[0, positions_to_mask])
                error = torch.mean(logit_diff).item()

            else:  # l2
                # L2 difference in logits at masked positions
                logit_diff = (logits_t[0, positions_to_mask] - logits_t0[0, positions_to_mask]) ** 2
                error = torch.mean(logit_diff).item()

            # Check for numerical issues
            if not np.isfinite(error):
                print(f"Warning: Non-finite error {error}")
                return 0.0

            # Optional normalization
            if self.normalize_output and error != 0.0:
                # Simple normalization by number of masked tokens
                error = error / len(positions_to_mask)

            # Return negative error (lower error = higher membership)
            return -abs(error)  # Use abs to ensure we're returning negative of positive error

        except Exception as e:
            print(f"Error in metric computation: {e}")
            return 0.0

    def extract_features(self, batch):
        """Extract features for downstream analysis."""
        features = []
        for text in batch["text"]:
            score = self._compute_pia_score(text)
            features.append(score)
        return np.array(features).reshape(-1, 1)