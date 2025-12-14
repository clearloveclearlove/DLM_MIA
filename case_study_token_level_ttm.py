#!/usr/bin/env python
"""
Token-Level Trajectory Turbulence Metric (TL-TTM)

Key Improvement:
  Instead of averaging probabilities across all tokens (sequence-level),
  we track individual token probabilities as context is progressively revealed.

Algorithm:
  1. Randomly select n target tokens to monitor
  2. Fix them as [MASK] throughout
  3. Progressively unmask the rest of context (100% → 0%)
  4. For each target token, compute its probability trajectory
  5. Calculate TTM for each token trajectory
  6. Aggregate (mean/max/min) to get final score

Hypothesis:
  - Member tokens show laminar flow (smooth probability increase as context reveals)
  - Non-member tokens show turbulent flow (oscillating, uncertain probability)
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
import pywt
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


def load_json_dataset(train_path: str, test_path: str) -> Dataset:
    """Load dataset from JSON files."""
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
    shuffle(combined)
    all_texts, all_labels = zip(*combined)

    return Dataset.from_dict({"text": list(all_texts), "label": list(all_labels)})


def calculate_ttm(trajectory: np.ndarray, wavelet: str = 'db1') -> float:
    """Calculate Trajectory Turbulence Metric."""
    if len(trajectory) < 4:
        return 0.0

    trajectory_centered = trajectory - trajectory.mean()

    try:
        coeffs = pywt.wavedec(trajectory_centered, wavelet=wavelet, level=1)
        cA, cDs = coeffs[0], coeffs[1:]
    except:
        return 0.0

    E_laminar = np.sum(np.square(cA))
    E_turbulent = sum(np.sum(np.square(d)) for d in cDs)

    total_energy = E_turbulent + E_laminar
    return float(E_turbulent / total_energy) if total_energy > 0 else 0.0


class TokenLevelTTMTester:
    """Token-Level TTM for membership inference."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
        self.dataset = load_json_dataset(args.json_train_path, args.json_test_path)

        members = self.dataset.filter(lambda x: x['label'] == 1)
        nonmembers = self.dataset.filter(lambda x: x['label'] == 0)

        logging.info(f"Original dataset: {len(members)} members, {len(nonmembers)} non-members")

        if args.num_samples > 0:
            samples_per_class = args.num_samples // 2
            if len(members) > samples_per_class:
                members = members.select(np.random.choice(len(members), samples_per_class, replace=False))
            if len(nonmembers) > samples_per_class:
                nonmembers = nonmembers.select(np.random.choice(len(nonmembers), samples_per_class, replace=False))

        from datasets import concatenate_datasets
        self.dataset = concatenate_datasets([members, nonmembers]).shuffle(seed=args.seed)

        logging.info(f"Final dataset: {len(members)} members, {len(nonmembers)} non-members")

        # Generate context mask schedule
        self.context_mask_ratios = np.linspace(1.0, 0.0, args.num_steps)
        logging.info(
            f"Context mask schedule ({args.num_steps} steps): {[f'{r:.2f}' for r in self.context_mask_ratios]}")
        logging.info(f"Target tokens per sample: {args.num_target_tokens}")
        logging.info(f"Aggregation method: {args.aggregation}")

        self.results = {
            'member_token_trajectories': [],
            'nonmember_token_trajectories': [],
            'member_ttm_scores': [],
            'nonmember_ttm_scores': [],
            'member_aggregated_ttm': [],
            'nonmember_aggregated_ttm': [],
            'metadata': {
                'model_path': args.model_path,
                'num_samples': len(self.dataset),
                'num_members': len(members),
                'num_nonmembers': len(nonmembers),
                'num_steps': args.num_steps,
                'num_target_tokens': args.num_target_tokens,
                'aggregation': args.aggregation,
                'wavelet': args.wavelet,
                'timestamp': datetime.now().isoformat()
            }
        }

    def compute_token_level_trajectories(self, text):
        """
        Compute probability trajectories for individual target tokens.

        Process:
          1. Tokenize text
          2. Randomly select n target tokens (keep them masked)
          3. Create context mask schedule (gradually unmask context)
          4. For each step, compute probability of each target token
          5. Return n trajectories (one per target token)
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

        if valid_len < self.args.num_target_tokens + 2:  # Need some context
            return None, None

        # Select target token positions (avoid padding and special tokens)
        valid_positions = torch.arange(1, valid_len - 1, device=self.device)  # Skip [CLS] and [SEP]

        if len(valid_positions) < self.args.num_target_tokens:
            return None, None

        # Randomly select target tokens
        perm = torch.randperm(len(valid_positions), device=self.device)
        target_indices = valid_positions[perm[: self.args.num_target_tokens]]
        target_indices = target_indices.sort()[0]  # Sort for consistency

        # Get ground truth tokens
        target_token_ids = input_ids[target_indices]

        # Get context positions (all positions except targets)
        all_positions = torch.arange(valid_len, device=self.device)
        context_mask = torch.ones(valid_len, dtype=torch.bool, device=self.device)
        context_mask[target_indices] = False
        context_positions = all_positions[context_mask]

        # Shuffle context positions for progressive unmasking
        context_perm = torch.randperm(len(context_positions), device=self.device)
        context_positions_shuffled = context_positions[context_perm]

        # Compute trajectories
        token_trajectories = [[] for _ in range(len(target_indices))]

        for context_mask_ratio in self.context_mask_ratios:
            # Determine how many context tokens to mask
            num_context_to_mask = int(np.round(context_mask_ratio * len(context_positions)))

            # Create masked input
            masked_ids = input_ids.clone()

            # Always mask target tokens
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
                    logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

                # Get probabilities
                probs = F.softmax(logits[0], dim=-1)

                # Extract probability for each target token
                for i, (pos, token_id) in enumerate(zip(target_indices, target_token_ids)):
                    token_prob = probs[pos, token_id].item()
                    token_trajectories[i].append(token_prob)

        # Convert to numpy arrays
        token_trajectories = [np.array(traj) for traj in token_trajectories]

        return token_trajectories, target_indices.cpu().numpy()

    def aggregate_ttm_scores(self, ttm_scores, method='mean'):
        """Aggregate multiple TTM scores using specified method."""
        if len(ttm_scores) == 0:
            return 0.0

        ttm_array = np.array(ttm_scores)

        if method == 'mean':
            return float(np.mean(ttm_array))
        elif method == 'max':
            return float(np.max(ttm_array))
        elif method == 'min':
            return float(np.min(ttm_array))
        elif method == 'median':
            return float(np.median(ttm_array))
        elif method == 'top_k':
            # Average of top-k most turbulent
            k = max(1, len(ttm_array) // 3)
            top_k = np.partition(ttm_array, -k)[-k:]
            return float(np.mean(top_k))
        else:
            return float(np.mean(ttm_array))

    def compute_all_samples(self):
        """Compute token-level TTM for all samples."""
        logging.info("Computing token-level TTM scores...")

        for i, sample in enumerate(tqdm(self.dataset, desc="Processing")):
            text = sample['text']
            label = sample['label']

            try:
                # Compute token-level trajectories
                token_trajs, target_indices = self.compute_token_level_trajectories(text)

                if token_trajs is None:
                    continue

                # Calculate TTM for each token
                token_ttm_scores = []
                for traj in token_trajs:
                    ttm = calculate_ttm(traj, wavelet=self.args.wavelet)
                    token_ttm_scores.append(ttm)

                # Aggregate TTM scores
                aggregated_ttm = self.aggregate_ttm_scores(token_ttm_scores, method=self.args.aggregation)

                if label == 1:  # Member
                    self.results['member_token_trajectories'].extend([traj.tolist() for traj in token_trajs])
                    self.results['member_ttm_scores'].extend(token_ttm_scores)
                    self.results['member_aggregated_ttm'].append(aggregated_ttm)
                else:  # Non-member
                    self.results['nonmember_token_trajectories'].extend([traj.tolist() for traj in token_trajs])
                    self.results['nonmember_ttm_scores'].extend(token_ttm_scores)
                    self.results['nonmember_aggregated_ttm'].append(aggregated_ttm)

            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue

        logging.info(f"Processed samples with {len(self.results['member_aggregated_ttm'])} members, "
                     f"{len(self.results['nonmember_aggregated_ttm'])} non-members")
        logging.info(f"Total token trajectories: {len(self.results['member_ttm_scores'])} member tokens, "
                     f"{len(self.results['nonmember_ttm_scores'])} non-member tokens")

    def analyze_results(self):
        """Statistical analysis."""
        # Token-level analysis
        member_token_ttm = np.array(self.results['member_ttm_scores'])
        nonmember_token_ttm = np.array(self.results['nonmember_ttm_scores'])

        # Sample-level analysis
        member_agg_ttm = np.array(self.results['member_aggregated_ttm'])
        nonmember_agg_ttm = np.array(self.results['nonmember_aggregated_ttm'])

        analysis = {}

        # Token-level statistics
        analysis['token_level'] = {
            'member_mean': float(member_token_ttm.mean()),
            'member_std': float(member_token_ttm.std()),
            'nonmember_mean': float(nonmember_token_ttm.mean()),
            'nonmember_std': float(nonmember_token_ttm.std()),
            'diff': float(nonmember_token_ttm.mean() - member_token_ttm.mean())
        }

        t_stat, p_value = stats.ttest_ind(member_token_ttm, nonmember_token_ttm)
        analysis['token_level']['ttest_pvalue'] = float(p_value)
        analysis['token_level']['ttest_significant'] = bool(p_value < 0.05)

        pooled_std = np.sqrt((member_token_ttm.std() ** 2 + nonmember_token_ttm.std() ** 2) / 2)
        cohens_d = (nonmember_token_ttm.mean() - member_token_ttm.mean()) / (pooled_std + 1e-8)
        analysis['token_level']['cohens_d'] = float(cohens_d)

        # Sample-level statistics
        analysis['sample_level'] = {
            'member_mean': float(member_agg_ttm.mean()),
            'member_std': float(member_agg_ttm.std()),
            'nonmember_mean': float(nonmember_agg_ttm.mean()),
            'nonmember_std': float(nonmember_agg_ttm.std()),
            'diff': float(nonmember_agg_ttm.mean() - member_agg_ttm.mean())
        }

        t_stat, p_value = stats.ttest_ind(member_agg_ttm, nonmember_agg_ttm)
        analysis['sample_level']['ttest_pvalue'] = float(p_value)
        analysis['sample_level']['ttest_significant'] = bool(p_value < 0.05)

        pooled_std = np.sqrt((member_agg_ttm.std() ** 2 + nonmember_agg_ttm.std() ** 2) / 2)
        cohens_d = (nonmember_agg_ttm.mean() - member_agg_ttm.mean()) / (pooled_std + 1e-8)
        analysis['sample_level']['cohens_d'] = float(cohens_d)

        # ROC-AUC
        all_agg_ttm = np.concatenate([member_agg_ttm, nonmember_agg_ttm])
        all_labels = np.concatenate([np.ones(len(member_agg_ttm)), np.zeros(len(nonmember_agg_ttm))])

        roc_auc = roc_auc_score(all_labels, -all_agg_ttm)  # Lower TTM = member
        analysis['roc_auc'] = float(roc_auc)

        analysis['hypothesis_verified'] = bool(
            analysis['sample_level']['nonmember_mean'] > analysis['sample_level']['member_mean'] and
            analysis['sample_level']['ttest_significant']
        )

        self.results['analysis'] = analysis
        return analysis

    def plot_results(self):
        """Generate visualizations."""
        member_trajs = np.array(self.results['member_token_trajectories'])
        nonmember_trajs = np.array(self.results['nonmember_token_trajectories'])
        member_agg = np.array(self.results['member_aggregated_ttm'])
        nonmember_agg = np.array(self.results['nonmember_aggregated_ttm'])
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
        ax.set_title('Token-Level Probability Trajectories', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

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

        # 3. Aggregated TTM distribution
        ax = axes[0, 2]
        ax.hist(member_agg, bins=30, alpha=0.6, label='Member', color='blue', density=True)
        ax.hist(nonmember_agg, bins=30, alpha=0.6, label='Non-member', color='red', density=True)
        ax.axvline(member_agg.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Member mean: {member_agg.mean():.4f}')
        ax.axvline(nonmember_agg.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Non-member mean: {nonmember_agg.mean():.4f}')
        ax.set_xlabel(f'Aggregated TTM ({self.args.aggregation})', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Sample-Level TTM Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 4. Box plot
        ax = axes[1, 0]
        data = [member_agg, nonmember_agg]
        bp = ax.boxplot(data, tick_labels=['Member', 'Non-member'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Aggregated TTM', fontsize=12)
        ax.set_title(f'TTM Comparison ({self.args.aggregation})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        p_value = analysis['sample_level']['ttest_pvalue']
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(1.5, max(member_agg.max(), nonmember_agg.max()) * 0.95,
                f'p = {p_value:.4f} {significance}', fontsize=11, ha='center')

        # 5. ROC Curve
        ax = axes[1, 1]
        all_agg = np.concatenate([member_agg, nonmember_agg])
        all_labels = np.concatenate([np.ones(len(member_agg)), np.zeros(len(nonmember_agg))])

        fpr, tpr, _ = roc_curve(all_labels, -all_agg)
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'TL-TTM (AUC={analysis["roc_auc"]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 6. Summary
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = "Token-Level TTM Results\n" + "=" * 50 + "\n\n"
        summary_text += f"Configuration:\n"
        summary_text += f"  Target tokens: {self.args.num_target_tokens}\n"
        summary_text += f"  Aggregation:    {self.args.aggregation}\n"
        summary_text += f"  Wavelet:       {self.args.wavelet}\n\n"

        summary_text += f"Sample-Level Results:\n"
        summary_text += f"  Member TTM:       {analysis['sample_level']['member_mean']:.4f} ± {analysis['sample_level']['member_std']:.4f}\n"
        summary_text += f"  Non-member TTM:   {analysis['sample_level']['nonmember_mean']:.4f} ± {analysis['sample_level']['nonmember_std']:.4f}\n"
        summary_text += f"  P-value:         {analysis['sample_level']['ttest_pvalue']:.6f}\n"
        summary_text += f"  Cohen's d:       {analysis['sample_level']['cohens_d']:.3f}\n"
        summary_text += f"  ROC-AUC:         {analysis['roc_auc']:.4f}\n\n"

        summary_text += f"Token-Level Results:\n"
        summary_text += f"  Member tokens:   {analysis['token_level']['member_mean']:.4f}\n"
        summary_text += f"  Non-member:       {analysis['token_level']['nonmember_mean']:.4f}\n"
        summary_text += f"  P-value:         {analysis['token_level']['ttest_pvalue']:.6f}\n"

        if analysis['hypothesis_verified']:
            conclusion = "✓ Hypothesis SUPPORTED"
        else:
            conclusion = "✗ Hypothesis REJECTED"

        summary_text += f"\n{conclusion}"

        ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round',
                                              facecolor='lightgreen' if analysis[
                                                  'hypothesis_verified'] else 'lightcoral',
                                              alpha=0.3))

        plt.tight_layout()
        save_path = self.output_dir / 'token_level_ttm_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot to:  {save_path}")
        plt.close()

    def save_results(self):
        """Save results to JSON."""
        output_file = self.output_dir / 'token_level_ttm_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"Saved results to: {output_file}")

    def print_summary(self):
        """Print summary."""
        analysis = self.results['analysis']

        print("\n" + "=" * 80)
        print("TOKEN-LEVEL TRAJECTORY TURBULENCE METRIC (TL-TTM) SUMMARY")
        print("=" * 80)

        print(f"\nConfiguration:")
        print(f"  Target tokens per sample: {self.args.num_target_tokens}")
        print(f"  Aggregation method:        {self.args.aggregation}")
        print(f"  Context mask steps:       {self.args.num_steps}")
        print(f"  Wavelet:                  {self.args.wavelet}")

        print("\n" + "-" * 80)
        print("SAMPLE-LEVEL RESULTS")
        print("-" * 80)

        print(f"\nAggregated TTM Scores:")
        print(
            f"  Member:        {analysis['sample_level']['member_mean']:.4f} ± {analysis['sample_level']['member_std']:.4f}")
        print(
            f"  Non-member:   {analysis['sample_level']['nonmember_mean']:.4f} ± {analysis['sample_level']['nonmember_std']:.4f}")
        print(f"  Difference:   {analysis['sample_level']['diff']:.4f}")
        print(f"  P-value:       {analysis['sample_level']['ttest_pvalue']:.6f}")
        print(f"  Cohen's d:     {analysis['sample_level']['cohens_d']:.3f}")
        print(f"  Significant:  {analysis['sample_level']['ttest_significant']}")

        print("\n" + "-" * 80)
        print("TOKEN-LEVEL RESULTS")
        print("-" * 80)

        print(f"\nIndividual Token TTM:")
        print(
            f"  Member tokens:     {analysis['token_level']['member_mean']:.4f} ± {analysis['token_level']['member_std']:.4f}")
        print(
            f"  Non-member tokens:  {analysis['token_level']['nonmember_mean']:.4f} ± {analysis['token_level']['nonmember_std']:.4f}")
        print(f"  Difference:        {analysis['token_level']['diff']:.4f}")
        print(f"  P-value:            {analysis['token_level']['ttest_pvalue']:.6f}")
        print(f"  Cohen's d:         {analysis['token_level']['cohens_d']:.3f}")

        print("\n" + "-" * 80)
        print("MEMBERSHIP INFERENCE PERFORMANCE")
        print("-" * 80)

        print(f"\nROC-AUC: {analysis['roc_auc']:.4f}")

        if analysis['hypothesis_verified']:
            print("\n✓ CONCLUSION: Token-level TTM hypothesis is SUPPORTED")
            print("  Non-member tokens show more turbulent trajectories")
        else:
            print("\n✗ CONCLUSION:  Hypothesis not fully supported")

        print("\n" + "=" * 80 + "\n")

    def run(self):
        """Run complete pipeline."""
        logging.info("Starting Token-Level TTM analysis...")
        self.compute_all_samples()
        logging.info("Performing analysis...")
        self.analyze_results()
        logging.info("Generating visualizations...")
        self.plot_results()
        self.save_results()
        self.print_summary()
        logging.info(f"✓ Complete!  Results in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Token-Level TTM for Membership Inference")

    parser.add_argument('--model_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/checkpoints/LLaDA-8B-Base-pretrained-pretraining-512-mimir-arxiv_32_5.0e-5_4epoch")
    parser.add_argument('--json_train_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/mimir-data/train/arxiv.json")
    parser.add_argument('--json_test_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/mimir-data/test/arxiv.json")

    # Token-level TTM parameters
    parser.add_argument('--num_target_tokens', type=int, default=10,
                        help="Number of target tokens to monitor per sample")
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'min', 'median', 'top_k'],
                        help="How to aggregate token-level TTM scores")
    parser.add_argument('--wavelet', type=str, default='db1')
    parser.add_argument('--num_steps', type=int, default=11,
                        help="Number of context unmasking steps")

    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default="./token_level_ttm_output")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    tester = TokenLevelTTMTester(args)
    tester.run()


if __name__ == '__main__':
    main()