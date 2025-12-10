# https://arxiv.org/pdf/2404.02936v1
import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def min_k_plusplus_diffusion(
        model: PreTrainedModel,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int = 20,
        mc_num: int = 100,
        mask_id: int = 126336
) -> np.ndarray:
    """
    Computes the min-k%++ attack score for a diffusion-based language model.

    Args:
        model: Diffusion-based model (e.g., LLaDA) returning logits.
        token_ids: [batch_size, seq_len] tensor of input token IDs.
        attention_mask: [batch_size, seq_len] tensor, 1 for real tokens, 0 for padding.
        k: Number of smallest log probabilities to sum.
        mc_num: Number of Monte Carlo passes for random masking.
        mask_id: Token ID used for masking.

    Returns:
        [batch_size] numpy array of scores (sum of k smallest average log p_correct values).
    """
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    sub_batch_size = 8  # Adjust based on GPU memory
    k_min_sums_list = []

    for i in range(0, batch_size, sub_batch_size):
        end_idx = min(i + sub_batch_size, batch_size)
        batch_tokens = token_ids[i:end_idx]
        batch_attention = attention_mask[i:end_idx]

        # List to collect log p_correct for each token across passes
        log_p_correct_list = [[] for _ in range((end_idx - i) * seq_len)]

        # Perform Monte Carlo passes
        for _ in range(mc_num):
            # Define prefix_mask (padding positions cannot be masked)
            prefix_mask = (batch_attention == 0)  # [sub_batch_size, seq_len]

            # Sample masking probability p for each sequence
            eps = 1e-3
            p = (1.0 - eps) * torch.rand(end_idx - i, device=device) + eps  # [sub_batch_size]

            # Determine masked positions
            maskable = ~prefix_mask
            rand = torch.rand(end_idx - i, seq_len, device=device)
            masked_positions = (rand < p.unsqueeze(1)) & maskable  # [sub_batch_size, seq_len]

            # Create noisy sequence
            noisy_seq = batch_tokens.clone()
            noisy_seq[masked_positions] = mask_id

            # Compute model logits
            with torch.no_grad():
                outputs = model(noisy_seq)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[
                    0]  # [sub_batch_size, seq_len, vocab_size]

            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # [sub_batch_size, seq_len, vocab_size]

            # Extract log p_correct for masked positions
            seq_indices, pos_indices = torch.where(masked_positions)
            if len(seq_indices) > 0:
                true_tokens = batch_tokens[seq_indices, pos_indices]
                log_p_correct = log_probs[seq_indices, pos_indices, true_tokens]

                # Store log p_correct for each masked token
                for seq_idx, pos_idx, log_p_val in zip(seq_indices, pos_indices, log_p_correct):
                    token_idx = seq_idx.item() * seq_len + pos_idx.item()
                    log_p_correct_list[token_idx].append(log_p_val.item())

        # Compute average log p_correct per token
        avg_log_p_correct = []
        for j in range(end_idx - i):
            seq_avg_log_p = []
            for m in range(seq_len):
                token_idx = j * seq_len + m
                if log_p_correct_list[token_idx]:
                    seq_avg_log_p.append(np.mean(log_p_correct_list[token_idx]))
                else:
                    # Rare case: token never masked; assign large negative value
                    seq_avg_log_p.append(-1000.0)
            avg_log_p_correct.append(seq_avg_log_p)

        # Compute score: sum of k smallest average log p_correct values
        k_min_sums = []
        for j in range(end_idx - i):
            # Filter to real tokens
            real_tokens = batch_attention[j] == 1
            seq_avg_log_p = [avg_log_p_correct[j][m] for m in range(seq_len) if real_tokens[m]]
            if len(seq_avg_log_p) >= k:
                sorted_log_p = sorted(seq_avg_log_p)  # Ascending order (most negative first)
                k_min_sum = sum(sorted_log_p[:k])
            else:
                # If fewer than k tokens, sum all and pad with -1000.0
                k_min_sum = sum(seq_avg_log_p) + (-1000.0 * (k - len(seq_avg_log_p)))
            # Ensure finite score
            k_min_sum = max(k_min_sum, -1e6)  # Cap to prevent extreme values
            k_min_sums.append(k_min_sum)

        k_min_sums_list.append(torch.tensor(k_min_sums, device=device))

        # Clean up GPU memory
        del batch_tokens, batch_attention, noisy_seq, outputs, logits, log_probs
        torch.cuda.empty_cache()

    return torch.cat(k_min_sums_list, dim=0).cpu().numpy()


class MinkplusplusAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.vocab_size = model.module.config.vocab_size if hasattr(model, 'module') else model.config.vocab_size
        self.batch_size = config.get('batch_size', 32)
        self.mc_num = config.get('mc_num', 4)  # Number of MC passes
        self.mask_id = config.get('mask_id', 126336)  # Mask token ID

    def run(self, dataset: Dataset) -> Dataset:
        all_scores = []

        # Process dataset in chunks to avoid memory issues
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            scores = self.score(batch)
            all_scores.extend(scores[self.name])

            # Clean up memory
            torch.cuda.empty_cache()

        return Dataset.from_dict({
            'text': dataset['text'],
            'label': dataset['label'],
            self.name: all_scores
        })

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding="longest",
            truncation=True,
            max_length=512  # Add explicit max length
        )

        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        k_min_probas = min_k_plusplus_diffusion(
            self.model,
            token_ids,
            attention_mask,
            k=self.config.get('k', 20),
            mc_num=self.mc_num,
            mask_id=self.mask_id
        )

        # Clean up GPU memory
        del token_ids, attention_mask, tokenized
        torch.cuda.empty_cache()

        return {self.name: k_min_probas}

    def extract_features(self, batch):
        # Note: This method is kept as is since it’s not directly used in the diffusion adaptation
        k = self.config['k']
        texts = batch["text"]
        features = []

        for text in texts:
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens)
                losses = torch.nn.functional.cross_entropy(
                    outputs.logits[:, :-1, :].flatten(0, 1),
                    tokens[:, 1:].flatten(),
                    reduction='none'
                ).reshape(tokens.shape[0], -1)

            top_k_losses = torch.topk(losses[0], min(k, losses.shape[1])).values
            bottom_k_losses = torch.topk(losses[0], min(k, losses.shape[1]), largest=False).values

            if len(top_k_losses) < k:
                top_k_losses = torch.cat([
                    top_k_losses,
                    torch.zeros(k - len(top_k_losses), device=self.device)
                ])
            if len(bottom_k_losses) < k:
                bottom_k_losses = torch.cat([
                    bottom_k_losses,
                    torch.zeros(k - len(bottom_k_losses), device=self.device)
                ])

            combined_features = torch.cat([top_k_losses, bottom_k_losses])
            features.append(combined_features.cpu().numpy())

        return np.array(features)