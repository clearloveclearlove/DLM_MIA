# https://arxiv.org/pdf/2310.16789v3
from attacks import AbstractAttack
from datasets import Dataset

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def min_k_prob_diffusion(
        model: PreTrainedModel,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int = 20,
        mc_num: int = 100,
        mask_id: int = 126336
) -> np.ndarray:
    """
    Computes the min-k% attack score for a diffusion-based language model.

    Args:
        model: Diffusion-based model (e.g., LLaDA) returning logits.
        token_ids: [batch_size, seq_len] tensor of input token IDs.
        attention_mask: [batch_size, seq_len] tensor, 1 for real tokens, 0 for padding.
        k: Number of smallest probabilities to sum.
        mc_num: Number of Monte Carlo passes for random masking.
        mask_id: Token ID used for masking.

    Returns:
        [batch_size] numpy array of scores (sum of k smallest average p_correct values).
    """
    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    # List to collect p_correct for each token across passes
    p_correct_list = [[] for _ in range(batch_size * seq_len)]

    # Perform Monte Carlo passes
    for _ in range(mc_num):
        # Define prefix_mask (padding positions cannot be masked)
        prefix_mask = (attention_mask == 0)  # [batch_size, seq_len]

        # Sample masking probability p for each sequence
        eps = 1e-3
        p = (1.0 - eps) * torch.rand(batch_size, device=device) + eps  # [batch_size]

        # Determine masked positions
        maskable = ~prefix_mask
        rand = torch.rand(batch_size, seq_len, device=device)
        masked_positions = (rand < p.unsqueeze(1)) & maskable  # [batch_size, seq_len]

        # Create noisy sequence
        noisy_seq = token_ids.clone()
        noisy_seq[masked_positions] = mask_id

        # Compute model logits
        with torch.no_grad():
            outputs = model(noisy_seq)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]

        # Extract p_correct for masked positions
        seq_indices, pos_indices = torch.where(masked_positions)
        if len(seq_indices) > 0:
            true_tokens = token_ids[seq_indices, pos_indices]
            p_correct = probs[seq_indices, pos_indices, true_tokens]

            # Store p_correct for each masked token
            for seq_idx, pos_idx, p_val in zip(seq_indices, pos_indices, p_correct):
                token_idx = seq_idx.item() * seq_len + pos_idx.item()
                p_correct_list[token_idx].append(p_val.item())

    # Compute average p_correct per token
    avg_p_correct = []
    for i in range(batch_size):
        seq_avg_p = []
        for j in range(seq_len):
            token_idx = i * seq_len + j
            if p_correct_list[token_idx]:
                seq_avg_p.append(np.mean(p_correct_list[token_idx]))
            else:
                # Rare case: token never masked; assign 0 (conservative)
                seq_avg_p.append(0.0)
        avg_p_correct.append(seq_avg_p)

    # Compute score: sum of k smallest average p_correct values
    k_min_sums = []
    for i in range(batch_size):
        # Filter to real tokens
        real_tokens = attention_mask[i] == 1
        seq_avg_p = [avg_p_correct[i][j] for j in range(seq_len) if real_tokens[j]]
        if len(seq_avg_p) >= k:
            sorted_p = sorted(seq_avg_p)  # Ascending order
            k_min_sum = sum(sorted_p[:k])
        else:
            k_min_sum = sum(seq_avg_p)  # Sum all if fewer than k
        k_min_sums.append(k_min_sum)

    return np.array(k_min_sums)


class MinkAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.mc_num = config.get('mc_num', 4)  # Number of MC passes
        self.mask_id = config.get('mask_id', 126336)  # Mask token ID

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.get('max_length', 512),
        )
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        # Compute scores using the diffusion-specific function
        k_min_sums = min_k_prob_diffusion(
            self.model,
            token_ids,
            attention_mask,
            k=self.config.get('k', 20),
            mc_num=self.mc_num,
            mask_id=self.mask_id
        )

        # Clear memory
        del token_ids, attention_mask
        torch.cuda.empty_cache()

        return {self.name: k_min_sums}

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config.get('batch_size', 4),
            new_fingerprint=f"{self.signature(dataset)}_v3",
        )
        return dataset