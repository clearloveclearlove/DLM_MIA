#!/usr/bin/env python
"""
Case Study: Verify PPL Trajectory Hypothesis for Membership Inference

Hypothesis to test:
  H1: Member samples show faster PPL drop (early drop) during denoising
  H2: Member samples have smaller trajectory AUC than non-members
  H3: The difference is statistically significant

This script:
  1. Loads target model and dataset
  2. Computes PPL trajectories for members and non-members
  3. Visualizes and compares trajectories
  4. Performs statistical tests
  5. Generates diagnostic plots
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from attack.misc.models import ModelManager
from attack.attacks.utils import get_model_nll_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_json_dataset(train_path: str, test_path: str) -> Dataset:
    """
    Load dataset from JSON files (same logic as attack/misc/dataset.py).

    Args:
        train_path: Path to training data (members, label=1)
        test_path: Path to test data (non-members, label=0)

    Returns:
        Dataset with 'text' and 'label' columns
    """
    # Load train subset (label 1) - handle both JSONL and nested JSON formats
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        try:
            # Try to parse as single JSON object first
            data = json.load(f)
            if 'data' in data and 'text' in data['data']:
                train_data = data['data']['text']
            else:
                # If it's a list of texts directly
                train_data = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # If that fails, try JSONL format (one JSON object per line)
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if 'text' in obj:
                            train_data.append(obj['text'])
                        else:
                            # If the object itself is the text
                            train_data.append(str(obj))
                    except json.JSONDecodeError:
                        continue

    train_labels = [1] * len(train_data)

    # Load test subset (label 0) - handle both JSONL and nested JSON formats
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        try:
            # Try to parse as single JSON object first
            data = json.load(f)
            if 'data' in data and 'text' in data['data']:
                test_data = data['data']['text']
            else:
                # If it's a list of texts directly
                test_data = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # If that fails, try JSONL format (one JSON object per line)
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if 'text' in obj:
                            test_data.append(obj['text'])
                        else:
                            # If the object itself is the text
                            test_data.append(str(obj))
                    except json.JSONDecodeError:
                        continue

    test_labels = [0] * len(test_data)

    # Combine and shuffle
    all_texts = train_data + test_data
    all_labels = train_labels + test_labels

    combined = list(zip(all_texts, all_labels))
    shuffle(combined)
    all_texts, all_labels = zip(*combined)

    return Dataset.from_dict({"text": list(all_texts), "label": list(all_labels)})


class TrajectoryHypothesisTester:
    """Test PPL trajectory hypothesis for membership inference."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Load model
        logging.info(f"Loading model from: {args.model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, _ = ModelManager.init_model(
            args.model_path, args.model_path, self.device
        )
        self.mask_id, self.shift_logits = get_model_nll_params(self.model)
        logging.info(f"Model loaded.  mask_id={self.mask_id}, shift_logits={self.shift_logits}")

        # Load dataset from JSON files
        logging.info(f"Loading dataset from JSON files:")
        logging.info(f"  Train (members): {args.json_train_path}")
        logging.info(f"  Test (non-members): {args.json_test_path}")

        self.dataset = load_json_dataset(args.json_train_path, args.json_test_path)

        # Verify dataset has required columns
        if 'text' not in self.dataset.column_names or 'label' not in self.dataset.column_names:
            raise ValueError(f"Dataset must have 'text' and 'label' columns.  Found: {self.dataset.column_names}")

        # Balance dataset (equal members and non-members)
        members = self.dataset.filter(lambda x: x['label'] == 1)
        nonmembers = self.dataset.filter(lambda x: x['label'] == 0)

        logging.info(f"Original dataset:  {len(members)} members, {len(nonmembers)} non-members")

        # Sampling
        if args.num_samples > 0:
            samples_per_class = args.num_samples // 2

            if len(members) > samples_per_class:
                member_indices = np.random.choice(len(members), samples_per_class, replace=False)
                members = members.select(member_indices)

            if len(nonmembers) > samples_per_class:
                nonmember_indices = np.random.choice(len(nonmembers), samples_per_class, replace=False)
                nonmembers = nonmembers.select(nonmember_indices)

        # Combine
        from datasets import concatenate_datasets
        self.dataset = concatenate_datasets([members, nonmembers])
        self.dataset = self.dataset.shuffle(seed=args.seed)

        logging.info(
            f"Final dataset: {len(members)} members, {len(nonmembers)} non-members (total: {len(self.dataset)})")

        # Generate mask schedule
        self.mask_ratios = np.linspace(1.0, 0.0, args.num_steps)
        logging.info(f"Mask schedule: {[f'{r:.2f}' for r in self.mask_ratios]}")

        # Storage for results
        self.results = {
            'member_trajectories': [],
            'nonmember_trajectories': [],
            'member_aucs': [],
            'nonmember_aucs': [],
            'member_labels': [],
            'nonmember_labels': [],
            'metadata': {
                'model_path': args.model_path,
                'json_train_path': args.json_train_path,
                'json_test_path': args.json_test_path,
                'num_samples': len(self.dataset),
                'num_members': len(members),
                'num_nonmembers': len(nonmembers),
                'num_steps': args.num_steps,
                'mask_ratios': self.mask_ratios.tolist(),
                'timestamp': datetime.now().isoformat()
            }
        }

    def compute_ppl_trajectory(self, text, num_runs=1):
        """
        Compute PPL trajectory for a single text sample.

        Returns:
            np.ndarray: PPL values at each denoising step, shape (num_steps,)
        """
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        valid_len = int(attention_mask.sum().item())

        if valid_len == 0:
            return np.ones(len(self.mask_ratios))

        # Sample fixed mask positions
        valid_pos = torch.arange(valid_len, device=self.device)
        perm = torch.randperm(valid_len, device=self.device)
        fixed_positions = valid_pos[perm]

        # Compute PPL at each step
        trajectory = []

        for mask_ratio in self.mask_ratios:
            step_ppls = []

            for _ in range(num_runs):
                # Create masked input
                masked_ids = input_ids.clone()
                num_to_mask = int(np.round(mask_ratio * valid_len))

                if num_to_mask > 0:
                    positions_to_mask = fixed_positions[:num_to_mask]
                    masked_ids[0, positions_to_mask] = self.mask_id

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=masked_ids,
                        attention_mask=attention_mask if not self.shift_logits else None
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    if self.shift_logits:
                        logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

                    # Compute NLL
                    ce = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        input_ids.view(-1),
                        reduction='none'
                    )

                    # Average over valid tokens
                    valid_ce = ce[attention_mask.view(-1)]
                    nll = valid_ce.mean().item()
                    # ppl = np.exp(nll)
                    step_ppls.append(nll)

            trajectory.append(np.mean(step_ppls))
        trajectory = trajectory/np.max(trajectory)
        return np.array(trajectory)

    def compute_all_trajectories(self):
        """Compute trajectories for all samples in dataset."""
        logging.info("Computing PPL trajectories for all samples...")

        for i, sample in enumerate(tqdm(self.dataset, desc="Processing")):
            text = sample['text']
            label = sample['label']  # 1 = member, 0 = non-member

            try:
                trajectory = self.compute_ppl_trajectory(text, num_runs=self.args.num_runs)
                auc = np.trapz(trajectory, x=self.mask_ratios)

                if label == 1:  # Member
                    self.results['member_trajectories'].append(trajectory.tolist())
                    self.results['member_aucs'].append(auc)
                    self.results['member_labels'].append(label)
                else:  # Non-member
                    self.results['nonmember_trajectories'].append(trajectory.tolist())
                    self.results['nonmember_aucs'].append(auc)
                    self.results['nonmember_labels'].append(label)

            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue

        logging.info(f"Processed {len(self.results['member_trajectories'])} members, "
                     f"{len(self.results['nonmember_trajectories'])} non-members")

    def analyze_trajectories(self):
        """Perform statistical analysis on trajectories."""
        member_trajs = np.array(self.results['member_trajectories'])
        nonmember_trajs = np.array(self.results['nonmember_trajectories'])
        member_aucs = np.array(self.results['member_aucs'])
        nonmember_aucs = np.array(self.results['nonmember_aucs'])

        analysis = {}

        # 1. Mean trajectories
        analysis['member_mean_traj'] = member_trajs.mean(axis=0).tolist()
        analysis['member_std_traj'] = member_trajs.std(axis=0).tolist()
        analysis['nonmember_mean_traj'] = nonmember_trajs.mean(axis=0).tolist()
        analysis['nonmember_std_traj'] = nonmember_trajs.std(axis=0).tolist()

        # 2. AUC statistics
        analysis['member_auc_mean'] = float(member_aucs.mean())
        analysis['member_auc_std'] = float(member_aucs.std())
        analysis['nonmember_auc_mean'] = float(nonmember_aucs.mean())
        analysis['nonmember_auc_std'] = float(nonmember_aucs.std())
        analysis['auc_diff'] = float(nonmember_aucs.mean() - member_aucs.mean())

        # 3. Statistical tests
        # T-test for AUC difference
        t_stat, p_value = stats.ttest_ind(member_aucs, nonmember_aucs)
        analysis['ttest_statistic'] = float(t_stat)
        analysis['ttest_pvalue'] = float(p_value)
        analysis['ttest_significant'] = bool(p_value < 0.05)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((member_aucs.std() ** 2 + nonmember_aucs.std() ** 2) / 2)
        cohens_d = (member_aucs.mean() - nonmember_aucs.mean()) / (pooled_std + 1e-8)
        analysis['cohens_d'] = float(cohens_d)

        # 4. Early drop analysis
        # Compare PPL drop from 100% to 50% mask
        mid_idx = len(self.mask_ratios) // 2
        member_early_drop = member_trajs[:, 0] - member_trajs[:, mid_idx]
        nonmember_early_drop = nonmember_trajs[:, 0] - nonmember_trajs[:, mid_idx]

        analysis['member_early_drop_mean'] = float(member_early_drop.mean())
        analysis['nonmember_early_drop_mean'] = float(nonmember_early_drop.mean())
        analysis['early_drop_diff'] = float(member_early_drop.mean() - nonmember_early_drop.mean())

        # 5. ROC-AUC for membership inference
        all_aucs = np.concatenate([member_aucs, nonmember_aucs])
        all_labels = np.concatenate([
            np.ones(len(member_aucs)),
            np.zeros(len(nonmember_aucs))
        ])

        # Try both directions
        roc_auc_direct = roc_auc_score(all_labels, -all_aucs)  # Negative:  small AUC = member
        roc_auc_inverse = roc_auc_score(all_labels, all_aucs)  # Positive: large AUC = member

        analysis['roc_auc_direct'] = float(roc_auc_direct)
        analysis['roc_auc_inverse'] = float(roc_auc_inverse)
        analysis['best_roc_auc'] = float(max(roc_auc_direct, roc_auc_inverse))
        analysis['hypothesis_correct'] = bool(roc_auc_direct > 0.5)

        # 6. Hypothesis verification
        hypothesis_results = {
            'H1_early_drop': {
                'statement': 'Members show faster PPL drop in early stages',
                'member_early_drop': analysis['member_early_drop_mean'],
                'nonmember_early_drop': analysis['nonmember_early_drop_mean'],
                'verified': bool(analysis['early_drop_diff'] > 0),
                'confidence': 'high' if abs(analysis['early_drop_diff']) > 100 else 'low'
            },
            'H2_smaller_auc': {
                'statement': 'Members have smaller trajectory AUC',
                'member_auc': analysis['member_auc_mean'],
                'nonmember_auc': analysis['nonmember_auc_mean'],
                'verified': bool(analysis['member_auc_mean'] < analysis['nonmember_auc_mean']),
                'confidence': 'high' if analysis['ttest_significant'] else 'low'
            },
            'H3_statistical_significance': {
                'statement': 'The difference is statistically significant (p < 0.05)',
                'p_value': analysis['ttest_pvalue'],
                'verified': analysis['ttest_significant'],
                'effect_size': analysis['cohens_d']
            }
        }

        analysis['hypothesis_verification'] = hypothesis_results

        self.results['analysis'] = analysis

        return analysis

    def plot_trajectories(self):
        """Generate visualization plots."""
        member_trajs = np.array(self.results['member_trajectories'])
        nonmember_trajs = np.array(self.results['nonmember_trajectories'])
        analysis = self.results['analysis']

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 10)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Mean trajectories with confidence intervals
        ax = axes[0, 0]
        mask_percentages = self.mask_ratios * 100

        member_mean = np.array(analysis['member_mean_traj'])
        member_std = np.array(analysis['member_std_traj'])
        nonmember_mean = np.array(analysis['nonmember_mean_traj'])
        nonmember_std = np.array(analysis['nonmember_std_traj'])

        ax.plot(mask_percentages, member_mean, 'o-', label='Member', color='blue', linewidth=2)
        ax.fill_between(mask_percentages, member_mean - member_std, member_mean + member_std,
                        alpha=0.3, color='blue')

        ax.plot(mask_percentages, nonmember_mean, 's-', label='Non-member', color='red', linewidth=2)
        ax.fill_between(mask_percentages, nonmember_mean - nonmember_std, nonmember_mean + nonmember_std,
                        alpha=0.3, color='red')

        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('Mean PPL Trajectories (100% → 0% mask)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # 100% to 0%

        # 2. Individual trajectories (sample)
        ax = axes[0, 1]
        sample_size = min(50, len(member_trajs), len(nonmember_trajs))

        for i in range(sample_size):
            ax.plot(mask_percentages, member_trajs[i], alpha=0.3, color='blue', linewidth=0.5)
        for i in range(sample_size):
            ax.plot(mask_percentages, nonmember_trajs[i], alpha=0.3, color='red', linewidth=0.5)

        ax.plot([], [], color='blue', label='Member (sample)', linewidth=2)
        ax.plot([], [], color='red', label='Non-member (sample)', linewidth=2)

        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title(f'Individual Trajectories (n={sample_size} each)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # 3. AUC distribution
        ax = axes[0, 2]
        member_aucs = self.results['member_aucs']
        nonmember_aucs = self.results['nonmember_aucs']

        ax.hist(member_aucs, bins=30, alpha=0.6, label='Member', color='blue', density=True)
        ax.hist(nonmember_aucs, bins=30, alpha=0.6, label='Non-member', color='red', density=True)
        ax.axvline(np.mean(member_aucs), color='blue', linestyle='--', linewidth=2,
                   label=f'Member mean: {np.mean(member_aucs):.1f}')
        ax.axvline(np.mean(nonmember_aucs), color='red', linestyle='--', linewidth=2,
                   label=f'Non-member mean:  {np.mean(nonmember_aucs):.1f}')

        ax.set_xlabel('Trajectory AUC', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('AUC Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 4. Box plot comparison
        ax = axes[1, 0]
        data = [member_aucs, nonmember_aucs]
        bp = ax.boxplot(data, labels=['Member', 'Non-member'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel('Trajectory AUC', fontsize=12)
        ax.set_title('AUC Comparison (Box Plot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistical annotation
        p_value = analysis['ttest_pvalue']
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(1.5, max(max(member_aucs), max(nonmember_aucs)) * 0.95,
        f'p = {p_value:.4f} {significance}', fontsize = 11, ha = 'center')

        # 5. ROC Curve
        ax = axes[1, 1]
        all_aucs = np.concatenate([member_aucs, nonmember_aucs])
        all_labels = np.concatenate([np.ones(len(member_aucs)), np.zeros(len(nonmember_aucs))])

        # Try both directions
        fpr_direct, tpr_direct, _ = roc_curve(all_labels, -all_aucs)
        fpr_inverse, tpr_inverse, _ = roc_curve(all_labels, all_aucs)

        roc_auc_direct = analysis['roc_auc_direct']
        roc_auc_inverse = analysis['roc_auc_inverse']

        ax.plot(fpr_direct, tpr_direct, 'b-', linewidth=2,
                label=f'Score = -AUC (AUC={roc_auc_direct:.4f})')
        ax.plot(fpr_inverse, tpr_inverse, 'r--', linewidth=2,
                label=f'Score = AUC (AUC={roc_auc_inverse:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve for Membership Inference', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 6. Hypothesis verification summary
        ax = axes[1, 2]
        ax.axis('off')

        hypothesis = analysis['hypothesis_verification']
        summary_text = "Hypothesis Verification Summary\n" + "=" * 40 + "\n\n"

        for h_name, h_data in hypothesis.items():
            status = "✓ VERIFIED" if h_data['verified'] else "✗ REJECTED"
            summary_text += f"{h_name}:\n"
            summary_text += f"  {h_data['statement']}\n"
            summary_text += f"  Status: {status}\n\n"

        summary_text += f"\nOverall Results:\n"
        summary_text += f"  Best ROC-AUC: {analysis['best_roc_auc']:.4f}\n"
        summary_text += f"  Effect Size (Cohen's d): {analysis['cohens_d']:.3f}\n"
        summary_text += f"  P-value: {analysis['ttest_pvalue']:.4f}\n"

        if analysis['hypothesis_correct']:
            conclusion = "✓ Hypothesis SUPPORTED:\nMembers have SMALLER AUC"
        else:
            conclusion = "✗ Hypothesis REJECTED:\nMembers have LARGER AUC"

        summary_text += f"\n{conclusion}"

        ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # Save figure
        save_path = self.output_dir / 'trajectory_hypothesis_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot to:  {save_path}")
        plt.close()

    def save_results(self):
        """Save analysis results to JSON."""
        output_file = self.output_dir / 'trajectory_hypothesis_results.json'

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logging.info(f"Saved results to: {output_file}")

    def print_summary(self):
        """Print summary to console."""
        analysis = self.results['analysis']

        print("\n" + "=" * 80)
        print("TRAJECTORY HYPOTHESIS VERIFICATION SUMMARY")
        print("=" * 80)

        print(f"\nDataset: {self.args.json_train_path} / {self.args.json_test_path}")
        print(f"Model: {self.args.model_path}")
        print(f"Samples: {len(self.results['member_trajectories'])} members, "
              f"{len(self.results['nonmember_trajectories'])} non-members")

        print("\n" + "-" * 80)
        print("HYPOTHESIS TESTS")
        print("-" * 80)

        for h_name, h_data in analysis['hypothesis_verification'].items():
            status = "✓ VERIFIED" if h_data['verified'] else "✗ REJECTED"
            print(f"\n{h_name}:  {status}")
            print(f"  Statement: {h_data['statement']}")

            if 'member_early_drop' in h_data:
                print(f"  Member early drop: {h_data['member_early_drop']:.2f}")
                print(f"  Non-member early drop: {h_data['nonmember_early_drop']:.2f}")

            if 'member_auc' in h_data:
                print(f"  Member AUC: {h_data['member_auc']:.2f} ± {analysis['member_auc_std']:.2f}")
                print(f"  Non-member AUC:  {h_data['nonmember_auc']:.2f} ± {analysis['nonmember_auc_std']:.2f}")

            if 'p_value' in h_data:
                print(f"  P-value: {h_data['p_value']:.6f}")
                print(f"  Effect size (Cohen's d): {h_data['effect_size']:.3f}")

        print("\n" + "-" * 80)
        print("MEMBERSHIP INFERENCE PERFORMANCE")
        print("-" * 80)

        print(f"\nROC-AUC (Score = -AUC): {analysis['roc_auc_direct']:.4f}")
        print(f"ROC-AUC (Score = AUC): {analysis['roc_auc_inverse']:.4f}")
        print(f"Best ROC-AUC: {analysis['best_roc_auc']:.4f}")

        if analysis['hypothesis_correct']:
            print("\n✓ CONCLUSION: Hypothesis is SUPPORTED")
            print("  Members have smaller trajectory AUC (faster PPL drop)")
            print("  → Use score = -AUC in trajectory attack")
        else:
            print("\n✗ CONCLUSION:  Hypothesis is REJECTED")
            print("  Members have LARGER trajectory AUC (slower PPL drop)")
            print("  → Use score = AUC in trajectory attack (or switch models)")

        print("\n" + "=" * 80 + "\n")

    def run(self):
        """Run complete hypothesis testing pipeline."""
        logging.info("Starting trajectory hypothesis testing...")

        # Compute trajectories
        self.compute_all_trajectories()

        # Analyze
        logging.info("Performing statistical analysis...")
        self.analyze_trajectories()

        # Visualize
        logging.info("Generating plots...")
        self.plot_trajectories()

        # Save
        self.save_results()

        # Print summary
        self.print_summary()

        logging.info(f"✓ Analysis complete! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify PPL trajectory hypothesis for membership inference"
    )

    # Model and data
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home1/yibiao/code/DLM-MIA/checkpoints/LLaDA-8B-Base-pretrained-pretraining-512-mimir-arxiv_32_5.0e-5_4epoch",
        help="Path to target model"
    )
    parser.add_argument(
        '--json_train_path',
        type=str,
        default="/home1/yibiao/code/DLM-MIA/mimir-data/train/arxiv.json",
        help="Path to training data JSON (members, label=1)"
    )
    parser.add_argument(
        '--json_test_path',
        type=str,
        default="/home1/yibiao/code/DLM-MIA/mimir-data/test/arxiv.json",
        help="Path to test data JSON (non-members, label=0)"
    )

    # Sampling
    parser.add_argument(
        '--num_samples',
        type=int,
        default=500,
        help="Number of samples to test (0 = all, default: 500 for speed)"
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=11,
        help="Number of denoising steps"
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=1,
        help="Number of sampling runs per step (for variance reduction)"
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./trajectory_hypothesis_output",
        help="Output directory for results"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Run hypothesis testing
    tester = TrajectoryHypothesisTester(args)
    tester.run()


if __name__ == '__main__':
    main()