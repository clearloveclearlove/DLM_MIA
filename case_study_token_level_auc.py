#!/usr/bin/env python
"""
Token-Level Trajectory AUC for Membership Inference (K-Sensitivity Analysis)

Key Updates:
  1. K-Sensitivity: Evaluates Top-K and Bottom-K across multiple ratios (1% to 50%).
  2. Case Study: Prints BOTTOM 10 tokens (Hardest to predict) to observe signals.
  3. Optimized Aggregation: Automatically finds the best K ratio.
"""

import argparse
import json
import logging
import os
import sys
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from attack.misc.models import ModelManager
from attack.attacks.utils import get_model_nll_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_dataset(train_path: str, test_path: str, seed: int = 42) -> Dataset:
    """Load dataset from JSON files with deterministic shuffling."""
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            train_data = data['data']['text'] if 'data' in data and 'text' in data['data'] else (
                data if isinstance(data, list) else [data])
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line.strip())
                        train_data.append(obj['text'] if 'text' in obj else str(obj))
                    except:
                        continue

    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            test_data = data['data']['text'] if 'data' in data and 'text' in data['data'] else (
                data if isinstance(data, list) else [data])
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line.strip())
                        test_data.append(obj['text'] if 'text' in obj else str(obj))
                    except:
                        continue

    all_texts = train_data + test_data
    all_labels = [1] * len(train_data) + [0] * len(test_data)
    combined = list(zip(all_texts, all_labels))

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(combined)

    all_texts, all_labels = zip(*combined)

    return Dataset.from_dict({"text": list(all_texts), "label": list(all_labels)})


def calculate_trajectory_auc(trajectory: np.ndarray, x_values: np.ndarray = None) -> float:
    """Calculate Area Under the trajectory Curve."""
    if len(trajectory) < 2:
        return 0.0

    if x_values is None:
        x_values = np.arange(len(trajectory))

    # Using trapezoidal rule
    auc = np.trapz(trajectory, x=x_values)
    x_range = x_values[-1] - x_values[0]

    # Normalize by x_range to keep score roughly between 0 and 1
    normalized_auc = auc / x_range if abs(x_range) > 1e-8 else auc

    return float(normalized_auc)


class TokenLevelAUCTester:
    """Token-Level Trajectory AUC with binning strategy."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set all seeds
        set_all_seeds(args.seed)

        # Load model
        logging.info(f"Loading model from:  {args.model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, _ = ModelManager.init_model(
            args.model_path, args.model_path, self.device
        )
        self.mask_id, self.shift_logits = get_model_nll_params(self.model)
        logging.info(f"Model loaded.  mask_id={self.mask_id}, shift_logits={self.shift_logits}")

        # Load dataset
        logging.info(f"Loading dataset from JSON files")
        self.dataset = load_json_dataset(args.json_train_path, args.json_test_path, seed=args.seed)

        members = self.dataset.filter(lambda x: x['label'] == 1)
        nonmembers = self.dataset.filter(lambda x: x['label'] == 0)

        logging.info(f"Original dataset: {len(members)} members, {len(nonmembers)} non-members")

        if args.num_samples > 0:
            samples_per_class = args.num_samples // 2

            if len(members) > samples_per_class:
                rng = np.random.RandomState(args.seed)
                member_indices = rng.choice(len(members), samples_per_class, replace=False)
                members = members.select(member_indices.tolist())

            if len(nonmembers) > samples_per_class:
                rng = np.random.RandomState(args.seed + 1)
                nonmember_indices = rng.choice(len(nonmembers), samples_per_class, replace=False)
                nonmembers = nonmembers.select(nonmember_indices.tolist())

        from datasets import concatenate_datasets
        self.dataset = concatenate_datasets([members, nonmembers])

        # Pick Case Study Indices (2 members, 2 non-members)
        member_indices = [i for i, x in enumerate(self.dataset) if x['label'] == 1]
        nonmember_indices = [i for i, x in enumerate(self.dataset) if x['label'] == 0]

        self.case_study_indices = []
        if len(member_indices) >= 2:
            self.case_study_indices.extend(random.sample(member_indices, 2))
        if len(nonmember_indices) >= 2:
            self.case_study_indices.extend(random.sample(nonmember_indices, 2))

        logging.info(f"Final dataset: {len(members)} members, {len(nonmembers)} non-members")
        logging.info(f"Case study indices: {self.case_study_indices}")

        # Generate context mask schedule
        self.context_mask_ratios = np.linspace(1.0, 0.0, args.num_steps)
        logging.info(
            f"Context mask schedule ({args.num_steps} steps): {[f'{r:.2f}' for r in self.context_mask_ratios]}")
        logging.info(f"Number of bins: {args.num_bins}")

        self.results = {
            'member_token_trajectories': [],
            'nonmember_token_trajectories': [],
            'member_token_aucs': [],
            'nonmember_token_aucs': [],
            'member_sample_aucs_list': [],
            'nonmember_sample_aucs_list': [],
            'metadata': {
                'model_path': args.model_path,
                'num_samples': len(self.dataset),
                'timestamp': datetime.now().isoformat()
            }
        }

    def compute_token_level_trajectories_binned(self, text):
        """Compute probability trajectories with Random Binning."""
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].bool()
        valid_len = int(attention_mask.sum().item())

        if valid_len < 2:
            return None

        # Get all valid token positions (skip special tokens)
        all_valid_positions = list(range(1, valid_len - 1))

        # CRITICAL: Shuffle to preserve local context across bins
        random.shuffle(all_valid_positions)

        if len(all_valid_positions) == 0:
            return None

        # Divide into bins
        num_bins = min(self.args.num_bins, len(all_valid_positions))
        bin_size = len(all_valid_positions) // num_bins

        results_list = []

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
            target_indices = torch.tensor(bin_positions, device=self.device)
            target_token_ids = input_ids[target_indices]

            # Get context positions (all except current bin)
            context_positions = [p for p in range(valid_len) if p not in bin_positions]
            context_positions = torch.tensor(context_positions, device=self.device)

            # Shuffle context for progressive unmasking
            g = torch.Generator(device=self.device)
            g.manual_seed(self.args.seed + bin_idx)
            context_perm = torch.randperm(len(context_positions), generator=g, device=self.device)
            context_positions_shuffled = context_positions[context_perm]

            # Compute trajectories for this bin
            bin_trajectories = [[] for _ in range(len(target_indices))]

            for context_mask_ratio in self.context_mask_ratios:
                # Determine how many context tokens to mask
                num_context_to_mask = int(np.round(context_mask_ratio * len(context_positions)))

                masked_ids = input_ids.clone()
                masked_ids[target_indices] = self.mask_id  # Always mask target

                if num_context_to_mask > 0:
                    positions_to_mask = context_positions_shuffled[:num_context_to_mask]
                    masked_ids[positions_to_mask] = self.mask_id

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=masked_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0) if not self.shift_logits else None
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    if self.shift_logits:
                        logits = torch.cat([logits[:, :   1, :], logits[:, :-1, :]], dim=1)

                    probs = F.softmax(logits[0], dim=-1)

                    for i, (pos, token_id) in enumerate(zip(target_indices, target_token_ids)):
                        token_prob = probs[pos, token_id].item()
                        bin_trajectories[i].append(token_prob)

            # Decode tokens and store trajectory
            for i, token_id in enumerate(target_token_ids):
                token_str = self.tokenizer.decode([token_id])
                results_list.append((token_str, np.array(bin_trajectories[i])))

        return results_list

    def aggregate_scores_dynamic(self, scores_list_of_lists, method='mean', ratio=None):
        """
        Dynamically aggregate scores for a list of samples.
        Supports specific ratios for top_k/bottom_k.
        """
        aggregated_results = []

        for scores in scores_list_of_lists:
            if not scores:
                aggregated_results.append(0.0)
                continue

            arr = np.array(scores)

            if method == 'mean':
                val = np.mean(arr)
            elif method == 'max':
                val = np.max(arr)
            elif method == 'min':
                val = np.min(arr)
            elif method == 'median':
                val = np.median(arr)
            elif method == 'top_k':
                # Use ratio if provided, else default to 10%
                r = ratio if ratio is not None else 0.1
                k = max(1, int(len(arr) * r))
                top_k_vals = np.partition(arr, -k)[-k:]
                val = np.mean(top_k_vals)
            elif method == 'bottom_k':
                # Use ratio if provided, else default to 10%
                r = ratio if ratio is not None else 0.1
                k = max(1, int(len(arr) * r))
                bottom_k_vals = np.partition(arr, k - 1)[:k]
                val = np.mean(bottom_k_vals)
            else:
                val = np.mean(arr)

            aggregated_results.append(float(val))

        return np.array(aggregated_results)

    def print_case_study(self, label, text, token_data):
        """Print detailed observation focusing on BOTTOM (Hardest) tokens."""
        print("\n" + "#" * 80)
        print(f"CASE STUDY OBSERVER | Label: {'MEMBER' if label == 1 else 'NON-MEMBER'}")
        print("#" * 80)
        print(f"Text Preview: {text[:150]}...")
        print("-" * 80)
        print(f"{'Token':<15} | {'AUC':<8} | Trajectory (Mask 100% -> 0%)")
        print("-" * 80)

        # Calculate AUCs for sorting
        data_with_auc = []
        for tok, traj in token_data:
            auc = calculate_trajectory_auc(traj, self.context_mask_ratios)
            data_with_auc.append((tok, traj, auc))

        # Sort by AUC ASCENDING (Smallest First) -> Focus on Hardest Tokens
        data_with_auc.sort(key=lambda x: x[2], reverse=False)

        # Print Bottom 10
        for tok, traj, auc in data_with_auc[:10]:
            traj_str = " ".join([f"{p:.3f}" for p in traj])
            tok_display = tok.replace('\n', '\\n').strip()
            if not tok_display: tok_display = "[Space]"
            print(f"{tok_display[:15]:<15} | {auc:.6f} | {traj_str}")
        print("#" * 80 + "\n")

    def compute_all_samples(self):
        """Compute token-level trajectory AUC for all samples."""
        logging.info("Computing token-level trajectory AUCs with binning...")

        for i, sample in enumerate(tqdm(self.dataset, desc="Processing")):
            text = sample['text']
            label = sample['label']

            try:
                # Returns list of (token_str, trajectory_array)
                token_results = self.compute_token_level_trajectories_binned(text)

                if token_results is None:
                    continue

                # Case Study Trigger
                if i in self.case_study_indices:
                    self.print_case_study(label, text, token_results)

                token_aucs = []
                for _, traj in token_results:
                    auc = calculate_trajectory_auc(traj, x_values=self.context_mask_ratios)
                    token_aucs.append(auc)

                if label == 1:  # Member
                    self.results['member_token_trajectories'].extend([t[1].tolist() for t in token_results])
                    self.results['member_token_aucs'].extend(token_aucs)
                    self.results['member_sample_aucs_list'].append(token_aucs)
                else:  # Non-member
                    self.results['nonmember_token_trajectories'].extend([t[1].tolist() for t in token_results])
                    self.results['nonmember_token_aucs'].extend(token_aucs)
                    self.results['nonmember_sample_aucs_list'].append(token_aucs)

            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue

    def analyze_results(self):
        """Analyze results with K-sensitivity analysis."""

        member_token_auc = np.array(self.results['member_token_aucs'])
        nonmember_token_auc = np.array(self.results['nonmember_token_aucs'])

        print("\n" + "=" * 80)
        print("TOKEN-LEVEL TRAJECTORY AUC ANALYSIS")
        print("=" * 80)

        _, p_value = stats.ttest_ind(member_token_auc, nonmember_token_auc)
        print(f"Total Member Tokens:     {len(member_token_auc)}")
        print(f"Total Non-Member Tokens: {len(nonmember_token_auc)}")
        print(f"Member Mean AUC:         {member_token_auc.mean():.4f}")
        print(f"Non-Member Mean AUC:     {nonmember_token_auc.mean():.4f}")
        print(f"P-value (Token-Level):   {p_value:.6e}")

        # --- K-Sensitivity Analysis ---
        print("\n" + "-" * 80)
        print("K-VALUE SENSITIVITY ANALYSIS (Top-K vs Bottom-K)")
        print("-" * 80)
        print(f"{'Method':<20} | {'Ratio':<6} | {'M-Mean':<8} | {'NM-Mean':<8} | {'Diff':<8} | {'ROC-AUC':<8}")
        print("-" * 80)

        # Ratios to test: 1%, 5%, 10%, 20%, 30%, 40%, 50%
        ratios = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        best_auc = 0
        best_method_name = ""

        # Test standard methods
        standard_methods = ['mean', 'median', 'max', 'min']
        for method in standard_methods:
            m_agg = self.aggregate_scores_dynamic(self.results['member_sample_aucs_list'], method)
            nm_agg = self.aggregate_scores_dynamic(self.results['nonmember_sample_aucs_list'], method)

            all_scores = np.concatenate([m_agg, nm_agg])
            all_labels = np.concatenate([np.ones(len(m_agg)), np.zeros(len(nm_agg))])
            auc_score = roc_auc_score(all_labels, all_scores)

            print(
                f"{method:<20} | {'N/A':<6} | {m_agg.mean():.4f}   | {nm_agg.mean():.4f}   | {m_agg.mean() - nm_agg.mean():.4f}   | {auc_score:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score
                best_method_name = method
                self.results['member_aggregated_auc'] = m_agg.tolist()
                self.results['nonmember_aggregated_auc'] = nm_agg.tolist()
                self.args.aggregation = method

        print("-" * 80)

        # Test Top-K with different ratios
        for r in ratios:
            m_agg = self.aggregate_scores_dynamic(self.results['member_sample_aucs_list'], 'top_k', ratio=r)
            nm_agg = self.aggregate_scores_dynamic(self.results['nonmember_sample_aucs_list'], 'top_k', ratio=r)

            all_scores = np.concatenate([m_agg, nm_agg])
            all_labels = np.concatenate([np.ones(len(m_agg)), np.zeros(len(nm_agg))])
            auc_score = roc_auc_score(all_labels, all_scores)

            name = f"top_k ({int(r * 100)}%)"
            print(
                f"{name:<20} | {r:<6.2f} | {m_agg.mean():.4f}   | {nm_agg.mean():.4f}   | {m_agg.mean() - nm_agg.mean():.4f}   | {auc_score:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score
                best_method_name = name
                self.results['member_aggregated_auc'] = m_agg.tolist()
                self.results['nonmember_aggregated_auc'] = nm_agg.tolist()
                self.args.aggregation = f"top_k_{r}"

        print("-" * 80)

        # Test Bottom-K with different ratios
        for r in ratios:
            m_agg = self.aggregate_scores_dynamic(self.results['member_sample_aucs_list'], 'bottom_k', ratio=r)
            nm_agg = self.aggregate_scores_dynamic(self.results['nonmember_sample_aucs_list'], 'bottom_k', ratio=r)

            all_scores = np.concatenate([m_agg, nm_agg])
            all_labels = np.concatenate([np.ones(len(m_agg)), np.zeros(len(nm_agg))])
            auc_score = roc_auc_score(all_labels, all_scores)

            name = f"bottom_k ({int(r * 100)}%)"
            print(
                f"{name:<20} | {r:<6.2f} | {m_agg.mean():.4f}   | {nm_agg.mean():.4f}   | {m_agg.mean() - nm_agg.mean():.4f}   | {auc_score:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score
                best_method_name = name
                self.results['member_aggregated_auc'] = m_agg.tolist()
                self.results['nonmember_aggregated_auc'] = nm_agg.tolist()
                self.args.aggregation = f"bottom_k_{r}"

        print("-" * 80)
        print(f"Best Aggregation Method: {best_method_name} (ROC-AUC: {best_auc:.4f})")
        return best_auc

    def plot_results(self):
        """Generate visualizations for the BEST aggregation method found."""
        member_agg = np.array(self.results['member_aggregated_auc'])
        nonmember_agg = np.array(self.results['nonmember_aggregated_auc'])

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Aggregated AUC distribution
        ax = axes[0]
        ax.hist(member_agg, bins=30, alpha=0.6, label='Member', color='blue', density=True)
        ax.hist(nonmember_agg, bins=30, alpha=0.6, label='Non-member', color='red', density=True)
        ax.set_xlabel(f'Trajectory AUC ({self.args.aggregation})')
        ax.set_title(f'Score Distribution ({self.args.aggregation})', fontweight='bold')
        ax.legend()

        # 2. ROC Curve
        ax = axes[1]
        all_agg = np.concatenate([member_agg, nonmember_agg])
        all_labels = np.concatenate([np.ones(len(member_agg)), np.zeros(len(nonmember_agg))])
        fpr, tpr, _ = roc_curve(all_labels, all_agg)
        auc_score = roc_auc_score(all_labels, all_agg)

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend(loc='lower right')

        plt.tight_layout()
        save_path = self.output_dir / 'token_level_auc_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """Save results to JSON."""
        results_to_save = {
            'best_aggregation_method': self.args.aggregation,
            'member_aggregated_auc': self.results['member_aggregated_auc'],
            'nonmember_aggregated_auc': self.results['nonmember_aggregated_auc'],
            'metadata': self.results['metadata']
        }
        output_file = self.output_dir / 'token_level_auc_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logging.info(f"Saved results to:  {output_file}")

    def run(self):
        self.compute_all_samples()
        self.analyze_results()
        self.plot_results()
        self.save_results()
        logging.info(f"✓ Complete!    Results in:   {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Token-Level Trajectory AUC Analysis")
    parser.add_argument('--model_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/checkpoints/LLaDA-8B-Base-pretrained-pretraining-512-mimir-arxiv_32_5.0e-5_4epoch")
    parser.add_argument('--json_train_path', type=str, default="/home1/yibiao/code/DLM-MIA/mimir-data/train/arxiv.json")
    parser.add_argument('--json_test_path', type=str, default="/home1/yibiao/code/DLM-MIA/mimir-data/test/arxiv.json")
    parser.add_argument('--num_bins', type=int, default=10)
    parser.add_argument('--aggregation', type=str, default='bottom_k')  # Placeholder, script will check all
    parser.add_argument('--num_steps', type=int, default=11)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default="./token_auc_sensitivity_test")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    TokenLevelAUCTester(args).run()


if __name__ == '__main__':
    main()