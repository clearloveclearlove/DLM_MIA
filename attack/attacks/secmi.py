# https://arxiv.org/pdf/2302.01316
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
import json
import os
from datetime import datetime

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params


class SecmiAttack(AbstractAttack):
    """
    SecMI adapted for diffusion language models (like LLaDA).

    Core idea: Member samples have smaller masking prediction errors compared to hold-out samples.
    Instead of pixel reconstruction error, we use token prediction cross-entropy on masked positions.
    Masking ratio t serves as the analog to diffusion timestep.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # Core parameters
        self.num_steps = int(config.get("steps", 5))
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))

        # Masking schedule (equivalent to timesteps in image diffusion)
        self.min_mask_ratio = float(config.get("min_mask_ratio", 0.1))
        self.max_mask_ratio = float(config.get("max_mask_ratio", 0.8))
        self.mask_schedule = config.get("mask_schedule", "linear")  # "linear" or "exponential"

        # Inference method
        self.inference_method = config.get("inference_method", "statistical")  # "statistical" or "threshold"

        # Model-specific parameters
        if 'model_mask_id' in config and 'model_shift_logits' in config:
            self.mask_id = config['model_mask_id']
            self.shift_logits = config['model_shift_logits']
        else:
            self.mask_id, self.shift_logits = get_model_nll_params(self.model)

        # Set random seed for reproducible masking
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(
            f"SecMI initialized: steps={self.num_steps}, mask_ratio=[{self.min_mask_ratio}, {self.max_mask_ratio}], mask_id={self.mask_id}")

    def run(self, dataset: Dataset) -> Dataset:
        """Run SecMI attack on the dataset."""
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
        """Compute membership scores for a batch of texts."""
        batch_size = len(texts)

        # Tokenize batch
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        seq_len = input_ids.size(1)

        # Generate masking ratios (equivalent to timesteps)
        mask_ratios = self._generate_mask_schedule()

        # Store errors across all steps for each sample
        step_errors = [[] for _ in range(batch_size)]

        for step, mask_ratio in enumerate(mask_ratios):
            # Create deterministic masks for this step (fixed seed per sample for consistency)
            masks = self._create_deterministic_masks(input_ids, attention_mask, mask_ratio, step)

            # Apply masks and compute prediction errors
            masked_input_ids = input_ids.clone()
            masked_input_ids[masks] = self.mask_id

            # Get model predictions on masked inputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask if not self.shift_logits else None
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                # Handle logit shifting if needed
                if self.shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                # Compute cross-entropy loss only on masked positions
                ce_losses = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    input_ids.view(-1),
                    reduction='none'
                ).view(batch_size, seq_len)

                # Extract errors for masked positions only
                for b in range(batch_size):
                    mask_positions = masks[b]
                    if mask_positions.any():
                        masked_errors = ce_losses[b][mask_positions]
                        avg_error = masked_errors.mean().item()
                        step_errors[b].append(avg_error)
                    else:
                        step_errors[b].append(0.0)

        # Aggregate errors across steps to get final membership scores
        batch_scores = []
        for b in range(batch_size):
            if len(step_errors[b]) == 0:
                batch_scores.append(0.0)
            else:
                # Lower error -> higher membership score (member samples have lower errors)
                if self.inference_method == "statistical":
                    # Use inverse of mean error with step weighting (early steps weighted more)
                    weights = 1.0 / (np.arange(len(step_errors[b])) + 1)
                    weights = weights / weights.sum()
                    weighted_avg_error = np.average(step_errors[b], weights=weights)
                    score = -weighted_avg_error  # Negative because lower error = higher membership
                else:  # simple threshold
                    score = -np.mean(step_errors[b])

                batch_scores.append(float(score))

        return batch_scores

    def _generate_mask_schedule(self):
        """Generate masking ratios for different steps (analog to timesteps)."""
        if self.mask_schedule == "exponential":
            # Exponential schedule: more steps at lower ratios
            ratios = np.logspace(
                np.log10(self.min_mask_ratio),
                np.log10(self.max_mask_ratio),
                self.num_steps
            )
        else:  # linear
            ratios = np.linspace(self.min_mask_ratio, self.max_mask_ratio, self.num_steps)

        return ratios.tolist()

    def _create_deterministic_masks(self, input_ids, attention_mask, mask_ratio, step_seed):
        """
        Create deterministic masks for consistency (like DDIM deterministic sampling).
        Each sample gets a reproducible mask based on its content and the step.
        """
        batch_size, seq_len = input_ids.shape
        masks = torch.zeros_like(input_ids, dtype=torch.bool)

        for b in range(batch_size):
            # Get valid positions (where attention_mask is True)
            valid_positions = torch.where(attention_mask[b])[0]
            valid_len = len(valid_positions)

            if valid_len == 0:
                continue

            # Number of tokens to mask
            num_to_mask = max(1, int(mask_ratio * valid_len))
            num_to_mask = min(num_to_mask, valid_len)

            # Create deterministic mask using sample content + step as seed
            sample_seed = hash((tuple(input_ids[b].cpu().numpy()), step_seed)) % (2 ** 32)
            rng = np.random.RandomState(sample_seed)

            # Randomly select positions to mask (but deterministically per sample+step)
            mask_indices = rng.choice(valid_len, size=num_to_mask, replace=False)
            positions_to_mask = valid_positions[mask_indices]

            masks[b, positions_to_mask] = True

        return masks