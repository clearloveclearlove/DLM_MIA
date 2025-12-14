#!/usr/bin/env python
"""
Case Study: Trajectory Turbulence Metric (TTM) for Membership Inference

Hypothesis:
  - Member samples show "laminar flow" (smooth, monotonic probability increase)
  - Non-member samples show "turbulent flow" (oscillating, hesitant probability changes)

Core Idea:
  Use Wavelet Transform to decompose probability trajectories into:
    - Low-frequency (laminar): overall trend
    - High-frequency (turbulent): rapid oscillations

  TTM Score = E_turbulent / (E_laminar + E_turbulent)
  Higher TTM → More turbulent → More likely non-member
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from attack.misc.models import ModelManager
from attack.attacks.utils import get_model_nll_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_json_dataset(train_path: str, test_path: str) -> Dataset:
    """Load dataset from JSON files."""
    # Load train subset (label 1)
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if 'data' in data and 'text' in data['data']:
                train_data = data['data']['text']
            else:
                train_data = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if 'text' in obj:
                            train_data.append(obj['text'])
                        else:
                            train_data.append(str(obj))
                    except json.JSONDecodeError:
                        continue

    train_labels = [1] * len(train_data)

    # Load test subset (label 0)
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if 'data' in data and 'text' in data['data']:
                test_data = data['data']['text']
            else:
                test_data = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if 'text' in obj:
                            test_data.append(obj['text'])
                        else:
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


def calculate_ttm(trajectory: np.ndarray, wavelet: str = 'db1') -> float:
    """
    Calculate Trajectory Turbulence Metric (TTM) for a single probability trajectory.

    Args:
        trajectory: 1D array representing token probability changes over decoding steps
        wavelet:  Wavelet type for decomposition (default: 'db1' - Daubechies 1)

    Returns:
        float: TTM score [0, 1].  Higher score = more turbulent = more likely non-member
    """
    if len(trajectory) < 4:  # Need minimum length for wavelet decomposition
        return 0.0

    # 1. Preprocessing: Remove mean to focus on fluctuations
    trajectory_centered = trajectory - trajectory.mean()

    # 2. Core:  Discrete Wavelet Transform (DWT) to decompose signal
    # cA = low-frequency coefficients (laminar flow)
    # cDs = high-frequency coefficients list (turbulent flow)
    try:
        coeffs = pywt.wavedec(trajectory_centered, wavelet=wavelet, level=1)
        cA, cDs = coeffs[0], coeffs[1:]
    except Exception as e:
        logging.warning(f"Wavelet decomposition failed: {e}")
        return 0.0

    # 3. Calculate energy (sum of squared coefficients)
    E_laminar = np.sum(np.square(cA))
    E_turbulent = sum(np.sum(np.square(d)) for d in cDs)

    # 4. Calculate TTM score:  ratio of turbulent energy to total energy
    total_energy = E_turbulent + E_laminar
    if total_energy == 0:
        return 0.0

    ttm_score = E_turbulent / total_energy

    return float(ttm_score)


def calculate_additional_metrics(trajectory: np.ndarray) -> dict:
    """Calculate additional trajectory characteristics for comparison."""
    metrics = {}

    # 1. Variance (traditional metric - ignores temporal order)
    metrics['variance'] = float(np.var(trajectory))

    # 2. Monotonicity score (how consistently the trajectory increases/decreases)
    diffs = np.diff(trajectory)
    if len(diffs) > 0:
        # Proportion of steps with consistent direction
        increasing = (diffs > 0).sum()
        decreasing = (diffs < 0).sum()
        metrics['monotonicity'] = float(max(increasing, decreasing) / len(diffs))
    else:
        metrics['monotonicity'] = 0.0

    # 3. Number of direction changes (zero-crossings in derivative)
    if len(diffs) > 1:
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        metrics['direction_changes'] = int(sign_changes)
    else:
        metrics['direction_changes'] = 0

    # 4. Smoothness (average absolute second derivative)
    if len(trajectory) > 2:
        second_diffs = np.diff(trajectory, n=2)
        metrics['smoothness'] = float(np.mean(np.abs(second_diffs)))
    else:
        metrics['smoothness'] = 0.0

    return metrics


class TrajectoryTurbulenceTester:
    """Test TTM hypothesis for membership inference."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
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
        logging.info(f"Loading dataset from JSON files:")
        logging.info(f"  Train (members): {args.json_train_path}")
        logging.info(f"  Test (non-members): {args.json_test_path}")

        self.dataset = load_json_dataset(args.json_train_path, args.json_test_path)

        # Balance dataset
        members = self.dataset.filter(lambda x: x['label'] == 1)
        nonmembers = self.dataset.filter(lambda x: x['label'] == 0)

        logging.info(f"Original dataset: {len(members)} members, {len(nonmembers)} non-members")

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
            f"Final dataset: {len(members)} members, {len(nonmembers)} non-members (total:  {len(self.dataset)})")

        # Generate mask schedule
        self.mask_ratios = np.linspace(1.0, 0.0, args.num_steps)
        logging.info(f"Mask schedule ({args.num_steps} steps): {[f'{r:.2f}' for r in self.mask_ratios]}")
        logging.info(f"Wavelet type: {args.wavelet}")

        # Storage for results
        self.results = {
            'member_trajectories': [],
            'nonmember_trajectories': [],
            'member_ttm_scores': [],
            'nonmember_ttm_scores': [],
            'member_metrics': [],
            'nonmember_metrics': [],
            'metadata': {
                'model_path': args.model_path,
                'json_train_path': args.json_train_path,
                'json_test_path': args.json_test_path,
                'num_samples': len(self.dataset),
                'num_members': len(members),
                'num_nonmembers': len(nonmembers),
                'num_steps': args.num_steps,
                'wavelet': args.wavelet,
                'mask_ratios': self.mask_ratios.tolist(),
                'timestamp': datetime.now().isoformat()
            }
        }

    def compute_probability_trajectory(self, text, num_runs=1):
        """
        Compute probability trajectory for correct tokens during progressive denoising.

        Returns:
            np.ndarray: Probability values at each denoising step
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
            return np.zeros(len(self.mask_ratios))

        # Sample fixed mask positions
        valid_pos = torch.arange(valid_len, device=self.device)
        perm = torch.randperm(valid_len, device=self.device)
        fixed_positions = valid_pos[perm]

        # Compute probability at each step
        trajectory = []

        for mask_ratio in self.mask_ratios:
            step_probs = []

            for _ in range(num_runs):
                # Create masked input
                masked_ids = input_ids.clone()
                num_to_mask = int(np.round(mask_ratio * valid_len))

                if num_to_mask > 0:
                    positions_to_mask = fixed_positions[: num_to_mask]
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

                    # Get probabilities
                    probs = F.softmax(logits, dim=-1)

                    # Extract probability of correct tokens (average over valid positions)
                    valid_probs = []
                    for pos in range(valid_len):
                        correct_token = input_ids[0, pos].item()
                        token_prob = probs[0, pos, correct_token].item()
                        valid_probs.append(token_prob)

                    avg_prob = np.mean(valid_probs) if valid_probs else 0.0
                    step_probs.append(avg_prob)

            trajectory.append(np.mean(step_probs))

        return np.array(trajectory)

    def compute_all_trajectories(self):
        """Compute probability trajectories and TTM scores for all samples."""
        logging.info("Computing probability trajectories and TTM scores...")

        for i, sample in enumerate(tqdm(self.dataset, desc="Processing")):
            text = sample['text']
            label = sample['label']

            try:
                # Compute trajectory
                trajectory = self.compute_probability_trajectory(text, num_runs=self.args.num_runs)

                # Calculate TTM score
                ttm_score = calculate_ttm(trajectory, wavelet=self.args.wavelet)

                # Calculate additional metrics
                metrics = calculate_additional_metrics(trajectory)
                metrics['ttm'] = ttm_score

                if label == 1:  # Member
                    self.results['member_trajectories'].append(trajectory.tolist())
                    self.results['member_ttm_scores'].append(ttm_score)
                    self.results['member_metrics'].append(metrics)
                else:  # Non-member
                    self.results['nonmember_trajectories'].append(trajectory.tolist())
                    self.results['nonmember_ttm_scores'].append(ttm_score)
                    self.results['nonmember_metrics'].append(metrics)

            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue

        logging.info(f"Processed {len(self.results['member_trajectories'])} members, "
                     f"{len(self.results['nonmember_trajectories'])} non-members")

    def analyze_results(self):
        """Perform statistical analysis on TTM scores and other metrics."""
        member_ttm = np.array(self.results['member_ttm_scores'])
        nonmember_ttm = np.array(self.results['nonmember_ttm_scores'])

        analysis = {}

        # 1. TTM statistics
        analysis['member_ttm_mean'] = float(member_ttm.mean())
        analysis['member_ttm_std'] = float(member_ttm.std())
        analysis['nonmember_ttm_mean'] = float(nonmember_ttm.mean())
        analysis['nonmember_ttm_std'] = float(nonmember_ttm.std())
        analysis['ttm_diff'] = float(nonmember_ttm.mean() - member_ttm.mean())

        # 2. Statistical tests
        t_stat, p_value = stats.ttest_ind(member_ttm, nonmember_ttm)
        analysis['ttest_statistic'] = float(t_stat)
        analysis['ttest_pvalue'] = float(p_value)
        analysis['ttest_significant'] = bool(p_value < 0.05)

        # Effect size
        pooled_std = np.sqrt((member_ttm.std() ** 2 + nonmember_ttm.std() ** 2) / 2)
        cohens_d = (nonmember_ttm.mean() - member_ttm.mean()) / (pooled_std + 1e-8)
        analysis['cohens_d'] = float(cohens_d)

        # 3. ROC-AUC
        all_ttm = np.concatenate([member_ttm, nonmember_ttm])
        all_labels = np.concatenate([
            np.ones(len(member_ttm)),
            np.zeros(len(nonmember_ttm))
        ])

        # Higher TTM = more turbulent = non-member
        roc_auc = roc_auc_score(all_labels, -all_ttm)  # Negative:  low TTM = member
        analysis['roc_auc_ttm'] = float(roc_auc)

        # 4. Compare with other metrics
        metric_names = ['variance', 'monotonicity', 'direction_changes', 'smoothness']
        for metric_name in metric_names:
            member_vals = np.array([m[metric_name] for m in self.results['member_metrics']])
            nonmember_vals = np.array([m[metric_name] for m in self.results['nonmember_metrics']])

            all_vals = np.concatenate([member_vals, nonmember_vals])

            # Try both directions
            auc_direct = roc_auc_score(all_labels, -all_vals)  # Low value = member
            auc_inverse = roc_auc_score(all_labels, all_vals)  # High value = member

            analysis[f'{metric_name}_auc'] = float(max(auc_direct, auc_inverse))
            analysis[f'{metric_name}_member_mean'] = float(member_vals.mean())
            analysis[f'{metric_name}_nonmember_mean'] = float(nonmember_vals.mean())

        # 5. Hypothesis verification
        hypothesis_results = {
            'laminar_vs_turbulent': {
                'statement': 'Members show laminar flow (low TTM), non-members show turbulent flow (high TTM)',
                'member_ttm': analysis['member_ttm_mean'],
                'nonmember_ttm': analysis['nonmember_ttm_mean'],
                'verified': bool(analysis['nonmember_ttm_mean'] > analysis['member_ttm_mean']),
                'confidence': 'high' if analysis['ttest_significant'] else 'low',
                'p_value': analysis['ttest_pvalue'],
                'effect_size': analysis['cohens_d']
            },
            'ttm_vs_variance': {
                'statement': 'TTM outperforms variance (temporal order matters)',
                'ttm_auc': analysis['roc_auc_ttm'],
                'variance_auc': analysis['variance_auc'],
                'verified': bool(analysis['roc_auc_ttm'] > analysis['variance_auc']),
                'improvement': float(analysis['roc_auc_ttm'] - analysis['variance_auc'])
            }
        }

        analysis['hypothesis_verification'] = hypothesis_results

        self.results['analysis'] = analysis

        return analysis

    def plot_results(self):
        """Generate comprehensive visualization."""
        member_trajs = np.array(self.results['member_trajectories'])
        nonmember_trajs = np.array(self.results['nonmember_trajectories'])
        member_ttm = np.array(self.results['member_ttm_scores'])
        nonmember_ttm = np.array(self.results['nonmember_ttm_scores'])
        analysis = self.results['analysis']

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))

        mask_percentages = self.mask_ratios * 100

        # 1. Mean trajectories
        ax = axes[0, 0]
        member_mean = member_trajs.mean(axis=0)
        member_std = member_trajs.std(axis=0)
        nonmember_mean = nonmember_trajs.mean(axis=0)
        nonmember_std = nonmember_trajs.std(axis=0)

        ax.plot(mask_percentages, member_mean, 'o-', label='Member (Laminar)',
                color='blue', linewidth=2)
        ax.fill_between(mask_percentages, member_mean - member_std, member_mean + member_std,
                        alpha=0.3, color='blue')

        ax.plot(mask_percentages, nonmember_mean, 's-', label='Non-member (Turbulent)',
                color='red', linewidth=2)
        ax.fill_between(mask_percentages, nonmember_mean - nonmember_std, nonmember_mean + nonmember_std,
                        alpha=0.3, color='red')

        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Average Probability', fontsize=12)
        ax.set_title('Mean Probability Trajectories', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # 2. Example laminar trajectory
        ax = axes[0, 1]
        sample_idx = np.argmin(member_ttm)  # Most laminar member
        ax.plot(mask_percentages, member_trajs[sample_idx], 'o-', color='blue', linewidth=2)
        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Example Laminar Flow\n(Member, TTM={member_ttm[sample_idx]:.4f})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # 3. Example turbulent trajectory
        ax = axes[0, 2]
        sample_idx = np.argmax(nonmember_ttm)  # Most turbulent non-member
        ax.plot(mask_percentages, nonmember_trajs[sample_idx], 's-', color='red', linewidth=2)
        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Example Turbulent Flow\n(Non-member, TTM={nonmember_ttm[sample_idx]:.4f})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # 4. TTM distribution
        ax = axes[1, 0]
        ax.hist(member_ttm, bins=30, alpha=0.6, label='Member', color='blue', density=True)
        ax.hist(nonmember_ttm, bins=30, alpha=0.6, label='Non-member', color='red', density=True)
        ax.axvline(member_ttm.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Member mean: {member_ttm.mean():.4f}')
        ax.axvline(nonmember_ttm.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Non-member mean: {nonmember_ttm.mean():.4f}')
        ax.set_xlabel('TTM Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('TTM Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 5. Box plot
        ax = axes[1, 1]
        data = [member_ttm, nonmember_ttm]
        bp = ax.boxplot(data, tick_labels=['Member\n(Laminar)', 'Non-member\n(Turbulent)'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel('TTM Score', fontsize=12)
        ax.set_title('TTM Score Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        p_value = analysis['ttest_pvalue']
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(1.5, max(member_ttm.max(), nonmember_ttm.max()) * 0.95,
        f'p = {p_value:.4f} {significance}', fontsize = 11, ha = 'center')

        # 6. ROC Curve
        ax = axes[1, 2]
        all_ttm = np.concatenate([member_ttm, nonmember_ttm])
        all_labels = np.concatenate([np.ones(len(member_ttm)), np.zeros(len(nonmember_ttm))])

        fpr, tpr, _ = roc_curve(all_labels, -all_ttm)
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'TTM (AUC={analysis["roc_auc_ttm"]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve for Membership Inference', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 7. Metric comparison
        ax = axes[2, 0]
        metrics = ['TTM', 'Variance', 'Monotonicity', 'Direction\nChanges', 'Smoothness']
        aucs = [
            analysis['roc_auc_ttm'],
            analysis['variance_auc'],
            analysis['monotonicity_auc'],
            analysis['direction_changes_auc'],
            analysis['smoothness_auc']
        ]

        colors = ['blue' if auc == max(aucs) else 'gray' for auc in aucs]
        bars = ax.bar(metrics, aucs, color=colors, alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylabel('ROC-AUC', fontsize=12)
        ax.set_title('Metric Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0.3, 1.0])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10)

        # 8. Wavelet decomposition example (member)
        ax = axes[2, 1]
        sample_idx = len(member_trajs) // 2
        traj = member_trajs[sample_idx]
        traj_centered = traj - traj.mean()

        coeffs = pywt.wavedec(traj_centered, wavelet=self.args.wavelet, level=1)
        cA, cD = coeffs[0], coeffs[1]

        # Reconstruct components
        laminar = pywt.upcoef('a', cA, self.args.wavelet, level=1, take=len(traj))[:len(traj)]
        turbulent = pywt.upcoef('d', cD, self.args.wavelet, level=1, take=len(traj))[:len(traj)]

        x = np.arange(len(traj))
        ax.plot(x, traj_centered, 'k-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(x, laminar, 'b--', linewidth=2, label='Laminar (Low-freq)')
        ax.plot(x, turbulent, 'r--', linewidth=2, label='Turbulent (High-freq)')

        ax.set_xlabel('Denoising Step', fontsize=12)
        ax.set_ylabel('Centered Probability', fontsize=12)
        ax.set_title('Wavelet Decomposition (Member)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 9. Summary text
        ax = axes[2, 2]
        ax.axis('off')

        summary_text = "Hypothesis Verification\n" + "=" * 50 + "\n\n"

        for h_name, h_data in analysis['hypothesis_verification'].items():
            status = "✓ VERIFIED" if h_data['verified'] else "✗ REJECTED"
            summary_text += f"{h_name}:\n  {status}\n\n"

        summary_text += f"\nKey Results:\n"
        summary_text += f"  TTM AUC:         {analysis['roc_auc_ttm']:.4f}\n"
        summary_text += f"  Variance AUC:   {analysis['variance_auc']:.4f}\n"
        summary_text += f"  Improvement:    {analysis['roc_auc_ttm'] - analysis['variance_auc']:.4f}\n\n"
        summary_text += f"  Member TTM:      {analysis['member_ttm_mean']:.4f} ± {analysis['member_ttm_std']:.4f}\n"
        summary_text += f"  Non-member TTM: {analysis['nonmember_ttm_mean']:.4f} ± {analysis['nonmember_ttm_std']:.4f}\n"
        summary_text += f"  P-value:        {analysis['ttest_pvalue']:.6f}\n"
        summary_text += f"  Cohen's d:      {analysis['cohens_d']:.3f}\n"

        if analysis['hypothesis_verification']['laminar_vs_turbulent']['verified']:
            conclusion = "✓ Hypothesis SUPPORTED:\nMembers are more laminar"
        else:
            conclusion = "✗ Hypothesis REJECTED"

        summary_text += f"\n{conclusion}"

        ax.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen' if
            analysis['hypothesis_verification']['laminar_vs_turbulent']['verified']
            else 'lightcoral', alpha=0.3))

        plt.tight_layout()

        save_path = self.output_dir / 'trajectory_turbulence_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot to:  {save_path}")
        plt.close()

    def save_results(self):
        """Save analysis results to JSON."""
        output_file = self.output_dir / 'trajectory_turbulence_results.json'

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logging.info(f"Saved results to: {output_file}")

    def print_summary(self):
        """Print summary to console."""
        analysis = self.results['analysis']

        print("\n" + "=" * 80)
        print("TRAJECTORY TURBULENCE METRIC (TTM) VERIFICATION SUMMARY")
        print("=" * 80)

        print(f"\nDataset: {self.args.json_train_path} / {self.args.json_test_path}")
        print(f"Model: {self.args.model_path}")
        print(f"Samples: {len(self.results['member_trajectories'])} members, "
              f"{len(self.results['nonmember_trajectories'])} non-members")
        print(f"Wavelet: {self.args.wavelet}")

        print("\n" + "-" * 80)
        print("HYPOTHESIS TESTS")
        print("-" * 80)

        for h_name, h_data in analysis['hypothesis_verification'].items():
            status = "✓ VERIFIED" if h_data['verified'] else "✗ REJECTED"
            print(f"\n{h_name}:  {status}")
            print(f"  {h_data['statement']}")

            if 'member_ttm' in h_data:
                print(f"  Member TTM:      {h_data['member_ttm']:.4f}")
                print(f"  Non-member TTM: {h_data['nonmember_ttm']:.4f}")
                print(f"  P-value:        {h_data['p_value']:.6f}")
                print(f"  Effect size:    {h_data['effect_size']:.3f}")

            if 'ttm_auc' in h_data:
                print(f"  TTM AUC:        {h_data['ttm_auc']:.4f}")
                print(f"  Variance AUC:   {h_data['variance_auc']:.4f}")
                print(f"  Improvement:    {h_data['improvement']:.4f}")

        print("\n" + "-" * 80)
        print("PERFORMANCE COMPARISON")
        print("-" * 80)

        print(f"\nTTM Score:           AUC = {analysis['roc_auc_ttm']:.4f}")
        print(f"Variance:            AUC = {analysis['variance_auc']:.4f}")
        print(f"Monotonicity:       AUC = {analysis['monotonicity_auc']:.4f}")
        print(f"Direction Changes:   AUC = {analysis['direction_changes_auc']:.4f}")
        print(f"Smoothness:         AUC = {analysis['smoothness_auc']:.4f}")

        if analysis['hypothesis_verification']['laminar_vs_turbulent']['verified']:
            print("\n✓ CONCLUSION: TTM hypothesis is SUPPORTED")
            print("  Members show laminar flow (smooth, low TTM)")
            print("  Non-members show turbulent flow (oscillating, high TTM)")
        else:
            print("\n✗ CONCLUSION: TTM hypothesis is REJECTED")

        if analysis['hypothesis_verification']['ttm_vs_variance']['verified']:
            print("\n✓ TTM outperforms variance (temporal order matters! )")
        else:
            print("\n✗ TTM does not outperform variance")

        print("\n" + "=" * 80 + "\n")

    def run(self):
        """Run complete hypothesis testing pipeline."""
        logging.info("Starting Trajectory Turbulence Metric (TTM) hypothesis testing...")

        # Compute trajectories and TTM scores
        self.compute_all_trajectories()

        # Analyze
        logging.info("Performing statistical analysis...")
        self.analyze_results()

        # Visualize
        logging.info("Generating plots...")
        self.plot_results()

        # Save
        self.save_results()

        # Print summary
        self.print_summary()

        logging.info(f"✓ Analysis complete!  Results saved to:  {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Trajectory Turbulence Metric (TTM) hypothesis for membership inference"
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
        help="Path to training data JSON (members)"
    )
    parser.add_argument(
        '--json_test_path',
        type=str,
        default="/home1/yibiao/code/DLM-MIA/mimir-data/test/arxiv.json",
        help="Path to test data JSON (non-members)"
    )

    # TTM parameters
    parser.add_argument(
        '--wavelet',
        type=str,
        default='db1',
        help="Wavelet type for decomposition (default: db1 - Daubechies 1)"
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
        help="Number of sampling runs per step"
    )

    # Sampling
    parser.add_argument(
        '--num_samples',
        type=int,
        default=200,
        help="Number of samples to test (0 = all)"
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
        default="./trajectory_turbulence_output",
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
    tester = TrajectoryTurbulenceTester(args)
    tester.run()


if __name__ == '__main__':
    main()