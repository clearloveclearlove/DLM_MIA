#!/usr/bin/env python
"""
Token-Level Trajectory AUC for Membership Inference (with Binning Strategy)

Key Improvement:
  Instead of randomly sampling tokens, we divide all tokens into bins and process
  each bin separately. This ensures:
    - Complete coverage of all tokens
    - Controlled memory usage
    - Deterministic and reproducible results

Algorithm:
  1. Divide all valid tokens into N bins
  2. For each bin:
     - Fix tokens in this bin as [MASK]
     - Progressively unmask context (100% → 0%)
     - Compute probability trajectory for each token
  3. Aggregate trajectories across all bins
  4. Compute final membership score
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from random import shuffle

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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_dataset(train_path: str, test_path: str, seed: int = 42) -> Dataset:
    """Load dataset from JSON files with deterministic shuffling."""
    import random

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

    auc = np.trapz(trajectory, x=x_values)
    x_range = x_values[-1] - x_values[0]
    normalized_auc = auc / x_range if abs(x_range) > 1e-8 else auc

    return float(normalized_auc)


def calculate_additional_metrics(trajectory: np.ndarray) -> dict:
    """Calculate additional trajectory characteristics."""
    metrics = {}

    metrics['auc'] = calculate_trajectory_auc(trajectory)
    metrics['initial_prob'] = float(trajectory[0])
    metrics['final_prob'] = float(trajectory[-1])
    metrics['total_gain'] = float(trajectory[-1] - trajectory[0])

    mid_idx = len(trajectory) // 2
    metrics['early_gain'] = float(trajectory[mid_idx] - trajectory[0])
    metrics['late_gain'] = float(trajectory[-1] - trajectory[mid_idx])

    diffs = np.diff(trajectory)
    if len(diffs) > 0:
        increasing = (diffs > 0).sum()
        decreasing = (diffs < 0).sum()
        metrics['monotonicity'] = float(max(increasing, decreasing) / len(diffs))
    else:
        metrics['monotonicity'] = 0.0

    metrics['variance'] = float(np.var(trajectory))

    return metrics


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

        logging.info(f"Final dataset: {len(members)} members, {len(nonmembers)} non-members")

        # Generate context mask schedule
        self.context_mask_ratios = np.linspace(1.0, 0.0, args.num_steps)
        logging.info(
            f"Context mask schedule ({args.num_steps} steps): {[f'{r:.2f}' for r in self.context_mask_ratios]}")
        logging.info(f"Number of bins: {args.num_bins}")
        logging.info(f"Aggregation method: {args.aggregation}")

        self.results = {
            'member_token_trajectories': [],
            'nonmember_token_trajectories': [],
            'member_token_aucs': [],
            'nonmember_token_aucs': [],
            'member_aggregated_auc': [],
            'nonmember_aggregated_auc': [],
            'member_token_metrics': [],
            'nonmember_token_metrics': [],
            'member_num_tokens_per_sample': [],
            'nonmember_num_tokens_per_sample': [],
            'metadata': {
                'model_path': args.model_path,
                'num_samples': len(self.dataset),
                'num_members': len(members),
                'num_nonmembers': len(nonmembers),
                'num_steps': args.num_steps,
                'num_bins': args.num_bins,
                'aggregation': args.aggregation,
                'timestamp': datetime.now().isoformat()
            }
        }

    def compute_token_level_trajectories_binned(self, text):
        """
        Compute probability trajectories for tokens using binning strategy.

        Process:
          1. Tokenize and identify all valid tokens
          2. Divide tokens into num_bins bins
          3. For each bin:
             - Process tokens in this bin as targets
             - Compute trajectories
          4. Aggregate across all bins

        Returns:
            token_trajectories: List of np.ndarray
            total_tokens_tested: int
        """
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
            return None, 0

        # Get all valid token positions (skip special tokens)
        all_valid_positions = list(range(1, valid_len - 1))

        if len(all_valid_positions) == 0:
            return None, 0

        # Divide into bins
        num_bins = min(self.args.num_bins, len(all_valid_positions))
        bin_size = len(all_valid_positions) // num_bins

        all_token_trajectories = []

        # Process each bin
        for bin_idx in range(num_bins):
            start_idx = bin_idx * bin_size
            if bin_idx == num_bins - 1:
                # Last bin gets remaining tokens
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

                # Create masked input
                masked_ids = input_ids.clone()

                # Always mask target tokens (current bin)
                masked_ids[target_indices] = self.mask_id

                # Mask portion of context
                if num_context_to_mask > 0:
                    positions_to_mask = context_positions_shuffled[:num_context_to_mask]
                    masked_ids[positions_to_mask] = self.mask_id

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=masked_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0) if not self.shift_logits else None
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    if self.shift_logits:
                        logits = torch.cat([logits[:, :   1, :], logits[:, :-1, :]], dim=1)

                    # Get probabilities
                    probs = F.softmax(logits[0], dim=-1)

                    # Extract probability for each target token in this bin
                    for i, (pos, token_id) in enumerate(zip(target_indices, target_token_ids)):
                        token_prob = probs[pos, token_id].item()
                        bin_trajectories[i].append(token_prob)

            # Convert to numpy and add to all trajectories
            for traj in bin_trajectories:
                all_token_trajectories.append(np.array(traj))

        return all_token_trajectories, len(all_valid_positions)

    def aggregate_scores(self, scores, method='mean'):
        """Aggregate multiple scores."""
        if len(scores) == 0:
            return 0.0

        scores_array = np.array(scores)

        if method == 'mean':
            return float(np.mean(scores_array))
        elif method == 'max':
            return float(np.max(scores_array))
        elif method == 'min':
            return float(np.min(scores_array))
        elif method == 'median':
            return float(np.median(scores_array))
        elif method == 'top_k':
            k = max(1, len(scores_array) // 3)
            top_k = np.partition(scores_array, -k)[-k:]
            return float(np.mean(top_k))
        elif method == 'bottom_k':
            k = max(1, len(scores_array) // 3)
            bottom_k = np.partition(scores_array, k - 1)[:k]
            return float(np.mean(bottom_k))
        else:
            return float(np.mean(scores_array))

    def compute_all_samples(self):
        """Compute token-level trajectory AUC for all samples using binning."""
        logging.info("Computing token-level trajectory AUCs with binning strategy...")

        for i, sample in enumerate(tqdm(self.dataset, desc="Processing")):
            text = sample['text']
            label = sample['label']

            try:
                # Compute token-level trajectories using bins
                token_trajs, num_tokens = self.compute_token_level_trajectories_binned(text)

                if token_trajs is None:
                    continue

                # Calculate AUC and metrics for each token
                token_aucs = []
                token_metrics_list = []

                for traj in token_trajs:
                    # Calculate AUC
                    auc = calculate_trajectory_auc(traj, x_values=self.context_mask_ratios)
                    token_aucs.append(auc)

                    # Calculate additional metrics
                    metrics = calculate_additional_metrics(traj)
                    token_metrics_list.append(metrics)

                # Aggregate AUC scores
                aggregated_auc = self.aggregate_scores(token_aucs, method=self.args.aggregation)

                if label == 1:  # Member
                    self.results['member_token_trajectories'].extend([traj.tolist() for traj in token_trajs])
                    self.results['member_token_aucs'].extend(token_aucs)
                    self.results['member_aggregated_auc'].append(aggregated_auc)
                    self.results['member_token_metrics'].extend(token_metrics_list)
                    self.results['member_num_tokens_per_sample'].append(num_tokens)
                else:  # Non-member
                    self.results['nonmember_token_trajectories'].extend([traj.tolist() for traj in token_trajs])
                    self.results['nonmember_token_aucs'].extend(token_aucs)
                    self.results['nonmember_aggregated_auc'].append(aggregated_auc)
                    self.results['nonmember_token_metrics'].extend(token_metrics_list)
                    self.results['nonmember_num_tokens_per_sample'].append(num_tokens)

            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue

        logging.info(f"Processed {len(self.results['member_aggregated_auc'])} member samples, "
                     f"{len(self.results['nonmember_aggregated_auc'])} non-member samples")

        avg_tokens_member = np.mean(self.results['member_num_tokens_per_sample']) if self.results[
            'member_num_tokens_per_sample'] else 0
        avg_tokens_nonmember = np.mean(self.results['nonmember_num_tokens_per_sample']) if self.results[
            'nonmember_num_tokens_per_sample'] else 0

        logging.info(
            f"Average tokens per sample: {avg_tokens_member:.1f} (member), {avg_tokens_nonmember:.1f} (non-member)")
        logging.info(f"Total token trajectories: {len(self.results['member_token_aucs'])} member tokens, "
                     f"{len(self.results['nonmember_token_aucs'])} non-member tokens")

    def analyze_results(self):
        """Statistical analysis."""
        member_token_auc = np.array(self.results['member_token_aucs'])
        nonmember_token_auc = np.array(self.results['nonmember_token_aucs'])

        member_agg_auc = np.array(self.results['member_aggregated_auc'])
        nonmember_agg_auc = np.array(self.results['nonmember_aggregated_auc'])

        analysis = {}

        # Token-level statistics
        analysis['token_level'] = {
            'member_mean': float(member_token_auc.mean()),
            'member_std': float(member_token_auc.std()),
            'nonmember_mean': float(nonmember_token_auc.mean()),
            'nonmember_std': float(nonmember_token_auc.std()),
            'diff': float(member_token_auc.mean() - nonmember_token_auc.mean())
        }

        t_stat, p_value = stats.ttest_ind(member_token_auc, nonmember_token_auc)
        analysis['token_level']['ttest_pvalue'] = float(p_value)
        analysis['token_level']['ttest_significant'] = bool(p_value < 0.05)

        pooled_std = np.sqrt((member_token_auc.std() ** 2 + nonmember_token_auc.std() ** 2) / 2)
        cohens_d = (member_token_auc.mean() - nonmember_token_auc.mean()) / (pooled_std + 1e-8)
        analysis['token_level']['cohens_d'] = float(cohens_d)

        # Sample-level statistics
        analysis['sample_level'] = {
            'member_mean': float(member_agg_auc.mean()),
            'member_std': float(member_agg_auc.std()),
            'nonmember_mean': float(nonmember_agg_auc.mean()),
            'nonmember_std': float(nonmember_agg_auc.std()),
            'diff': float(member_agg_auc.mean() - nonmember_agg_auc.mean())
        }

        t_stat, p_value = stats.ttest_ind(member_agg_auc, nonmember_agg_auc)
        analysis['sample_level']['ttest_pvalue'] = float(p_value)
        analysis['sample_level']['ttest_significant'] = bool(p_value < 0.05)

        pooled_std = np.sqrt((member_agg_auc.std() ** 2 + nonmember_agg_auc.std() ** 2) / 2)
        cohens_d = (member_agg_auc.mean() - nonmember_agg_auc.mean()) / (pooled_std + 1e-8)
        analysis['sample_level']['cohens_d'] = float(cohens_d)

        # ROC-AUC
        all_agg_auc = np.concatenate([member_agg_auc, nonmember_agg_auc])
        all_sample_labels = np.concatenate([np.ones(len(member_agg_auc)), np.zeros(len(nonmember_agg_auc))])

        roc_auc_direct = roc_auc_score(all_sample_labels, -all_agg_auc)
        roc_auc_inverse = roc_auc_score(all_sample_labels, all_agg_auc)

        analysis['roc_auc_direct'] = float(roc_auc_direct)
        analysis['roc_auc_inverse'] = float(roc_auc_inverse)
        analysis['best_roc_auc'] = float(max(roc_auc_direct, roc_auc_inverse))
        analysis['use_negative_score'] = bool(roc_auc_direct > roc_auc_inverse)

        # Analyze other metrics
        all_token_labels = np.concatenate([
            np.ones(len(self.results['member_token_metrics'])),
            np.zeros(len(self.results['nonmember_token_metrics']))
        ])

        metric_names = ['initial_prob', 'final_prob', 'total_gain', 'early_gain', 'late_gain', 'monotonicity',
                        'variance']
        for metric_name in metric_names:
            member_vals = np.array([m[metric_name] for m in self.results['member_token_metrics']])
            nonmember_vals = np.array([m[metric_name] for m in self.results['nonmember_token_metrics']])

            all_vals = np.concatenate([member_vals, nonmember_vals])

            auc_d = roc_auc_score(all_token_labels, -all_vals)
            auc_i = roc_auc_score(all_token_labels, all_vals)

            analysis[f'{metric_name}_auc'] = float(max(auc_d, auc_i))
            analysis[f'{metric_name}_member_mean'] = float(member_vals.mean())
            analysis[f'{metric_name}_nonmember_mean'] = float(nonmember_vals.mean())

        analysis['hypothesis_verified'] = bool(analysis['sample_level']['ttest_significant'])

        self.results['analysis'] = analysis
        return analysis

    def plot_results(self):
        """Generate visualizations."""
        member_trajs = np.array(self.results['member_token_trajectories'])
        nonmember_trajs = np.array(self.results['nonmember_token_trajectories'])
        member_agg = np.array(self.results['member_aggregated_auc'])
        nonmember_agg = np.array(self.results['nonmember_aggregated_auc'])
        analysis = self.results['analysis']

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        context_mask_pct = self.context_mask_ratios * 100

        # 1. Mean token trajectories
        ax = axes[0, 0]
        member_mean = member_trajs.mean(axis=0)
        member_std = member_trajs.std(axis=0)
        nonmember_mean = nonmember_trajs.mean(axis=0)
        nonmember_std = nonmember_trajs.std(axis=0)

        ax.plot(context_mask_pct, member_mean, 'o-', label='Member Tokens', color='blue', linewidth=2)
        ax.fill_between(context_mask_pct, member_mean - member_std, member_mean + member_std, alpha=0.3, color='blue')

        ax.plot(context_mask_pct, nonmember_mean, 's-', label='Non-member Tokens', color='red', linewidth=2)
        ax.fill_between(context_mask_pct, nonmember_mean - nonmember_std, nonmember_mean + nonmember_std, alpha=0.3,
                        color='red')

        ax.set_xlabel('Context Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Token Probability', fontsize=12)
        ax.set_title(f'Token-Level Probability Trajectories\n(Binning: {self.args.num_bins} bins)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        ax.fill_between(context_mask_pct, 0, member_mean, alpha=0.1, color='blue', label='Member AUC')

        # 2. Sample trajectories
        ax = axes[0, 1]
        sample_size = min(30, len(member_trajs), len(nonmember_trajs))
        for i in range(sample_size):
            ax.plot(context_mask_pct, member_trajs[i], alpha=0.3, color='blue', linewidth=0.5)
            ax.plot(context_mask_pct, nonmember_trajs[i], alpha=0.3, color='red', linewidth=0.5)

        ax.plot([], [], color='blue', label=f'Member tokens (n={sample_size})', linewidth=2)
        ax.plot([], [], color='red', label=f'Non-member tokens (n={sample_size})', linewidth=2)
        ax.set_xlabel('Context Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Token Probability', fontsize=12)
        ax.set_title('Individual Token Trajectories', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # 3. Aggregated AUC distribution
        ax = axes[0, 2]
        ax.hist(member_agg, bins=30, alpha=0.6, label='Member', color='blue', density=True)
        ax.hist(nonmember_agg, bins=30, alpha=0.6, label='Non-member', color='red', density=True)
        ax.axvline(member_agg.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Member mean: {member_agg.mean():.4f}')
        ax.axvline(nonmember_agg.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Non-member mean: {nonmember_agg.mean():.4f}')
        ax.set_xlabel(f'Trajectory AUC ({self.args.aggregation})', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Sample-Level AUC Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. Box plot
        ax = axes[1, 0]
        data = [member_agg, nonmember_agg]
        bp = ax.boxplot(data, tick_labels=['Member', 'Non-member'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Trajectory AUC', fontsize=12)
        ax.set_title(f'AUC Comparison ({self.args.aggregation})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        p_value = analysis['sample_level']['ttest_pvalue']
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(1.5, max(member_agg.max(), nonmember_agg.max()) * 0.95,
        f'p = {p_value:.4f} {significance}', fontsize = 11, ha = 'center')

        # 5. ROC Curve
        ax = axes[1, 1]
        all_agg = np.concatenate([member_agg, nonmember_agg])
        all_labels = np.concatenate([np.ones(len(member_agg)), np.zeros(len(nonmember_agg))])

        fpr_d, tpr_d, _ = roc_curve(all_labels, -all_agg)
        fpr_i, tpr_i, _ = roc_curve(all_labels, all_agg)

        ax.plot(fpr_d, tpr_d, 'b-', linewidth=2,
                label=f'Score = -AUC (AUC={analysis["roc_auc_direct"]: .4f})')
        ax.plot(fpr_i, tpr_i, 'r--', linewidth=2,
                label=f'Score = AUC (AUC={analysis["roc_auc_inverse"]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 6. Metric comparison
        ax = axes[1, 2]
        metrics = ['Trajectory\nAUC', 'Total\nGain', 'Early\nGain', 'Late\nGain', 'Monotonicity']
        aucs = [
            analysis['best_roc_auc'],
            analysis['total_gain_auc'],
            analysis['early_gain_auc'],
            analysis['late_gain_auc'],
            analysis['monotonicity_auc']
        ]

        colors = ['blue' if auc == max(aucs) else 'gray' for auc in aucs]
        bars = ax.bar(metrics, aucs, color=colors, alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylabel('ROC-AUC', fontsize=12)
        ax.set_title('Metric Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0.4, max(0.8, max(aucs) + 0.05)])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / 'token_level_auc_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot to:    {save_path}")
        plt.close()

    def save_results(self):
        """Save results to JSON."""
        results_to_save = {
            'member_aggregated_auc': [float(x) for x in self.results['member_aggregated_auc']],
            'nonmember_aggregated_auc': [float(x) for x in self.results['nonmember_aggregated_auc']],
            'analysis': self.results['analysis'],
            'metadata': self.results['metadata']
        }

        output_file = self.output_dir / 'token_level_auc_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logging.info(f"Saved results to:  {output_file}")

    def print_summary(self):
        """Print summary."""
        analysis = self.results['analysis']

        print("\n" + "=" * 80)
        print("TOKEN-LEVEL TRAJECTORY AUC SUMMARY (WITH BINNING)")
        print("=" * 80)

        print(f"\nConfiguration:")
        print(f"  Number of bins:           {self.args.num_bins}")
        print(f"  Aggregation method:      {self.args.aggregation}")
        print(f"  Context mask steps:      {self.args.num_steps}")

        avg_tokens_member = np.mean(self.results['member_num_tokens_per_sample']) if self.results[
            'member_num_tokens_per_sample'] else 0
        avg_tokens_nonmember = np.mean(self.results['nonmember_num_tokens_per_sample']) if self.results[
            'nonmember_num_tokens_per_sample'] else 0
        print(f"  Avg tokens/sample:       {avg_tokens_member:.1f} (member), {avg_tokens_nonmember:.1f} (non-member)")

        print("\n" + "-" * 80)
        print("SAMPLE-LEVEL RESULTS")
        print("-" * 80)

        print(f"\nAggregated Trajectory AUC:")
        print(
            f"  Member:          {analysis['sample_level']['member_mean']:.4f} ± {analysis['sample_level']['member_std']:.4f}")
        print(
            f"  Non-member:   {analysis['sample_level']['nonmember_mean']:.4f} ± {analysis['sample_level']['nonmember_std']:.4f}")
        print(f"  Difference:   {analysis['sample_level']['diff']:.4f}")
        print(f"  P-value:       {analysis['sample_level']['ttest_pvalue']:.6f}")
        print(f"  Cohen's d:     {analysis['sample_level']['cohens_d']:.3f}")
        print(f"  Significant:  {analysis['sample_level']['ttest_significant']}")

        print("\n" + "-" * 80)
        print("TOKEN-LEVEL RESULTS")
        print("-" * 80)

        print(f"\nIndividual Token AUC:")
        print(
            f"  Member tokens:     {analysis['token_level']['member_mean']: .4f} ± {analysis['token_level']['member_std']:.4f}")
        print(
            f"  Non-member tokens:    {analysis['token_level']['nonmember_mean']:.4f} ± {analysis['token_level']['nonmember_std']:.4f}")
        print(f"  Difference:        {analysis['token_level']['diff']:.4f}")
        print(f"  P-value:              {analysis['token_level']['ttest_pvalue']:.6f}")
        print(f"  Cohen's d:         {analysis['token_level']['cohens_d']:.3f}")

        print("\n" + "-" * 80)
        print("MEMBERSHIP INFERENCE PERFORMANCE")
        print("-" * 80)

        print(f"\nROC-AUC (Score = -AUC): {analysis['roc_auc_direct']:.4f}")
        print(f"ROC-AUC (Score = AUC):  {analysis['roc_auc_inverse']:.4f}")
        print(f"Best ROC-AUC:           {analysis['best_roc_auc']:.4f}")

        if analysis['use_negative_score']:
            print("\n→ Use negative score (small trajectory AUC = member)")
        else:
            print("\n→ Use positive score (large trajectory AUC = member)")

        print("\n" + "-" * 80)
        print("OTHER METRICS")
        print("-" * 80)

        print(f"\nTotal Gain:     AUC = {analysis['total_gain_auc']:.4f}")
        print(f"Early Gain:     AUC = {analysis['early_gain_auc']:.4f}")
        print(f"Late Gain:      AUC = {analysis['late_gain_auc']:.4f}")
        print(f"Monotonicity:  AUC = {analysis['monotonicity_auc']:.4f}")

        if analysis['hypothesis_verified']:
            print("\n✓ CONCLUSION: Token-level trajectory AUC shows statistically significant difference")
        else:
            print("\n✗ CONCLUSION: No significant difference found")

        print("\n" + "=" * 80 + "\n")

    def run(self):
        """Run complete pipeline."""
        logging.info("Starting Token-Level Trajectory AUC analysis with binning...")
        self.compute_all_samples()
        logging.info("Performing analysis...")
        self.analyze_results()
        logging.info("Generating visualizations...")
        self.plot_results()
        self.save_results()
        self.print_summary()
        logging.info(f"✓ Complete!    Results in:   {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Token-Level Trajectory AUC with Binning")

    parser.add_argument('--model_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/checkpoints/LLaDA-8B-Base-pretrained-pretraining-512-mimir-arxiv_32_5.0e-5_4epoch")
    parser.add_argument('--json_train_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/mimir-data/train/arxiv.json")
    parser.add_argument('--json_test_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/mimir-data/test/arxiv.json")

    # Binning parameters
    parser.add_argument('--num_bins', type=int, default=10,
                        help="Number of bins to divide tokens into (default: 10)")
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'min', 'median', 'top_k', 'bottom_k'],
                        help="How to aggregate token-level AUC scores")
    parser.add_argument('--num_steps', type=int, default=11,
                        help="Number of context unmasking steps")

    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default="./token_level_auc_output")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    tester = TokenLevelAUCTester(args)
    tester.run()


if __name__ == '__main__':
    main()