#!/usr/bin/env python
"""
Leave-One-Out (LOO) Membership Inference Case Study (Optimized with Sampling)

Two Approaches:
  1. PPL-based (White-box):
     - Randomly sample and mask tokens one at a time
     - Compute probability of the ground truth token
     - Aggregate as sequence-level PPL
     - Hypothesis: Members have lower PPL

  2. Prediction-based (Black-box):
     - Randomly sample and mask tokens one at a time
     - Predict the most likely token
     - Check if prediction matches ground truth
     - Aggregate as accuracy/error rate
     - Hypothesis: Members have higher accuracy

Optimization:  Only test a random sample of tokens (controlled by --loo_sample_ratio)
to significantly speed up computation while maintaining statistical validity.
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


class LeaveOneOutTester:
    """Leave-One-Out analysis for membership inference."""

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
        logging.info(f"LOO sample ratio: {args.loo_sample_ratio:.0%}")

        self.results = {
            'member_ppl': [],
            'nonmember_ppl': [],
            'member_accuracy': [],
            'nonmember_accuracy': [],
            'member_top_k_accuracy': {k: [] for k in [1, 5, 10]},
            'nonmember_top_k_accuracy': {k: [] for k in [1, 5, 10]},
            'member_token_probs': [],
            'nonmember_token_probs': [],
            'member_prediction_confidence': [],
            'nonmember_prediction_confidence': [],
            'member_num_tokens_tested': [],
            'nonmember_num_tokens_tested': [],
            'metadata': {
                'model_path': args.model_path,
                'num_samples': len(self.dataset),
                'num_members': len(members),
                'num_nonmembers': len(nonmembers),
                'max_length': args.max_length,
                'loo_sample_ratio': args.loo_sample_ratio,
                'timestamp': datetime.now().isoformat()
            }
        }

    def compute_leave_one_out_metrics(self, text):
        """
        Compute LOO metrics for a single text (with token sampling).

        Only tests a random sample of tokens (controlled by loo_sample_ratio)
        to significantly speed up computation.

        Returns:
            dict with:
              - ppl: Sequence-level perplexity (estimated from sampled tokens)
              - accuracy: Prediction accuracy (top-1)
              - top_k_accuracy: Top-k accuracy for k in [1, 5, 10]
              - token_probs: List of ground truth token probabilities
              - prediction_confidence: List of top-1 prediction confidences
              - num_tokens_tested: Number of tokens actually tested
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

        if valid_len < 2:  # Need at least 2 tokens
            return None

        # Skip special tokens (CLS, SEP, PAD)
        valid_positions = list(range(1, valid_len - 1))  # Skip first and last token

        if len(valid_positions) == 0:
            return None

        # 🔥 Random sampling: only test a subset of tokens
        num_tokens_to_sample = max(1, int(len(valid_positions) * self.args.loo_sample_ratio))
        sampled_positions = np.random.choice(valid_positions, size=num_tokens_to_sample, replace=False)

        token_probs = []
        correct_predictions = []
        top_k_correct = {k: [] for k in [1, 5, 10]}
        prediction_confidences = []

        # Leave-one-out:  mask each sampled token and predict
        for pos in sampled_positions:
            # Create masked input (mask only current position)
            masked_ids = input_ids.clone()
            masked_ids[pos] = self.mask_id

            # Get ground truth token
            ground_truth = input_ids[pos].item()

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=masked_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0) if not self.shift_logits else None
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                if self.shift_logits:
                    logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

                # Get probabilities at masked position
                token_logits = logits[0, pos, :]
                probs = F.softmax(token_logits, dim=-1)

                # 1. PPL-based: probability of ground truth
                gt_prob = probs[ground_truth].item()
                token_probs.append(gt_prob)

                # 2. Prediction-based: check if prediction matches
                top_k_preds = torch.topk(probs, k=10)

                # Top-1 prediction
                predicted_token = top_k_preds.indices[0].item()
                pred_confidence = top_k_preds.values[0].item()
                prediction_confidences.append(pred_confidence)

                # Top-1 accuracy
                correct_predictions.append(int(predicted_token == ground_truth))

                # Top-k accuracy
                for k in [1, 5, 10]:
                    top_k_tokens = top_k_preds.indices[: k].cpu().numpy()
                    top_k_correct[k].append(int(ground_truth in top_k_tokens))

        # Aggregate metrics
        metrics = {}

        # 1. PPL-based
        # Compute PPL from token probabilities
        valid_probs = [p for p in token_probs if p > 0]
        if valid_probs:
            log_probs = [np.log(p) for p in valid_probs]
            avg_log_prob = np.mean(log_probs)
            ppl = np.exp(-avg_log_prob)
            metrics['ppl'] = float(ppl)
        else:
            metrics['ppl'] = float('inf')

        # 2. Prediction-based
        metrics['accuracy'] = float(np.mean(correct_predictions))
        metrics['top_k_accuracy'] = {k: float(np.mean(top_k_correct[k])) for k in [1, 5, 10]}

        # Additional metrics
        metrics['token_probs'] = token_probs
        metrics['prediction_confidence'] = prediction_confidences
        metrics['avg_token_prob'] = float(np.mean(token_probs)) if token_probs else 0.
        0
        metrics['avg_prediction_confidence'] = float(np.mean(prediction_confidences)) if prediction_confidences else 0.0
        metrics['num_tokens_tested'] = len(sampled_positions)

        return metrics

    def compute_all_samples(self):
        """Compute LOO metrics for all samples."""
        logging.info("Computing Leave-One-Out metrics...")

        for i, sample in enumerate(tqdm(self.dataset, desc="Processing")):
            text = sample['text']
            label = sample['label']

            try:
                metrics = self.compute_leave_one_out_metrics(text)

                if metrics is None:
                    continue

                if label == 1:  # Member
                    self.results['member_ppl'].append(metrics['ppl'])
                    self.results['member_accuracy'].append(metrics['accuracy'])
                    for k in [1, 5, 10]:
                        self.results['member_top_k_accuracy'][k].append(metrics['top_k_accuracy'][k])
                    self.results['member_token_probs'].extend(metrics['token_probs'])
                    self.results['member_prediction_confidence'].extend(metrics['prediction_confidence'])
                    self.results['member_num_tokens_tested'].append(metrics['num_tokens_tested'])
                else:  # Non-member
                    self.results['nonmember_ppl'].append(metrics['ppl'])
                    self.results['nonmember_accuracy'].append(metrics['accuracy'])
                    for k in [1, 5, 10]:
                        self.results['nonmember_top_k_accuracy'][k].append(metrics['top_k_accuracy'][k])
                    self.results['nonmember_token_probs'].extend(metrics['token_probs'])
                    self.results['nonmember_prediction_confidence'].extend(metrics['prediction_confidence'])
                    self.results['nonmember_num_tokens_tested'].append(metrics['num_tokens_tested'])

            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue

        logging.info(f"Processed {len(self.results['member_ppl'])} member samples, "
                     f"{len(self.results['nonmember_ppl'])} non-member samples")

        avg_tokens_member = np.mean(self.results['member_num_tokens_tested']) if self.results[
            'member_num_tokens_tested'] else 0
        avg_tokens_nonmember = np.mean(self.results['nonmember_num_tokens_tested']) if self.results[
            'nonmember_num_tokens_tested'] else 0
        logging.info(
            f"Average tokens tested:  {avg_tokens_member:.1f} (member), {avg_tokens_nonmember:.1f} (non-member)")

    def analyze_results(self):
        """Statistical analysis."""
        member_ppl = np.array(self.results['member_ppl'])
        nonmember_ppl = np.array(self.results['nonmember_ppl'])

        member_acc = np.array(self.results['member_accuracy'])
        nonmember_acc = np.array(self.results['nonmember_accuracy'])

        analysis = {}

        # 1. PPL-based analysis
        analysis['ppl'] = {
            'member_mean': float(member_ppl.mean()),
            'member_std': float(member_ppl.std()),
            'member_median': float(np.median(member_ppl)),
            'nonmember_mean': float(nonmember_ppl.mean()),
            'nonmember_std': float(nonmember_ppl.std()),
            'nonmember_median': float(np.median(nonmember_ppl)),
            'diff': float(nonmember_ppl.mean() - member_ppl.mean())
        }

        # Filter out inf values for statistical tests
        member_ppl_finite = member_ppl[np.isfinite(member_ppl)]
        nonmember_ppl_finite = nonmember_ppl[np.isfinite(nonmember_ppl)]

        if len(member_ppl_finite) > 0 and len(nonmember_ppl_finite) > 0:
            t_stat, p_value = stats.ttest_ind(member_ppl_finite, nonmember_ppl_finite)
            analysis['ppl']['ttest_pvalue'] = float(p_value)
            analysis['ppl']['ttest_significant'] = bool(p_value < 0.05)

            pooled_std = np.sqrt((member_ppl_finite.std() ** 2 + nonmember_ppl_finite.std() ** 2) / 2)
            cohens_d = (member_ppl_finite.mean() - nonmember_ppl_finite.mean()) / (pooled_std + 1e-8)
            analysis['ppl']['cohens_d'] = float(cohens_d)

            # ROC-AUC for PPL
            all_ppl_finite = np.concatenate([member_ppl_finite, nonmember_ppl_finite])
            all_labels = np.concatenate([np.ones(len(member_ppl_finite)), np.zeros(len(nonmember_ppl_finite))])

            roc_auc_ppl = roc_auc_score(all_labels, -all_ppl_finite)  # Negative:  low PPL = member
            analysis['ppl']['roc_auc'] = float(roc_auc_ppl)

        # 2. Accuracy-based analysis
        analysis['accuracy'] = {
            'member_mean': float(member_acc.mean()),
            'member_std': float(member_acc.std()),
            'nonmember_mean': float(nonmember_acc.mean()),
            'nonmember_std': float(nonmember_acc.std()),
            'diff': float(member_acc.mean() - nonmember_acc.mean())
        }

        t_stat, p_value = stats.ttest_ind(member_acc, nonmember_acc)
        analysis['accuracy']['ttest_pvalue'] = float(p_value)
        analysis['accuracy']['ttest_significant'] = bool(p_value < 0.05)

        pooled_std = np.sqrt((member_acc.std() ** 2 + nonmember_acc.std() ** 2) / 2)
        cohens_d = (member_acc.mean() - nonmember_acc.mean()) / (pooled_std + 1e-8)
        analysis['accuracy']['cohens_d'] = float(cohens_d)

        # ROC-AUC for accuracy
        all_acc = np.concatenate([member_acc, nonmember_acc])
        all_labels = np.concatenate([np.ones(len(member_acc)), np.zeros(len(nonmember_acc))])

        roc_auc_acc = roc_auc_score(all_labels, all_acc)  # High accuracy = member
        analysis['accuracy']['roc_auc'] = float(roc_auc_acc)

        # 3. Top-k accuracy analysis
        analysis['top_k_accuracy'] = {}
        for k in [1, 5, 10]:
            member_top_k = np.array(self.results['member_top_k_accuracy'][k])
            nonmember_top_k = np.array(self.results['nonmember_top_k_accuracy'][k])

            analysis['top_k_accuracy'][f'top_{k}'] = {
                'member_mean': float(member_top_k.mean()),
                'nonmember_mean': float(nonmember_top_k.mean()),
                'diff': float(member_top_k.mean() - nonmember_top_k.mean())
            }

            all_top_k = np.concatenate([member_top_k, nonmember_top_k])
            roc_auc_top_k = roc_auc_score(all_labels, all_top_k)
            analysis['top_k_accuracy'][f'top_{k}']['roc_auc'] = float(roc_auc_top_k)

        # 4. Token-level analysis
        member_token_probs = np.array(self.results['member_token_probs'])
        nonmember_token_probs = np.array(self.results['nonmember_token_probs'])

        analysis['token_level'] = {
            'member_avg_prob': float(member_token_probs.mean()),
            'nonmember_avg_prob': float(nonmember_token_probs.mean()),
            'diff': float(member_token_probs.mean() - nonmember_token_probs.mean())
        }

        # 5. Sampling statistics
        analysis['sampling'] = {
            'avg_tokens_tested_member': float(np.mean(self.results['member_num_tokens_tested'])) if self.results[
                'member_num_tokens_tested'] else 0,
            'avg_tokens_tested_nonmember': float(np.mean(self.results['nonmember_num_tokens_tested'])) if self.results[
                'nonmember_num_tokens_tested'] else 0,
            'sample_ratio': float(self.args.loo_sample_ratio)
        }

        # 6. Hypothesis verification
        analysis['hypothesis'] = {
            'ppl_hypothesis': {
                'statement': 'Members have lower PPL',
                'verified': bool(analysis['ppl'].get('roc_auc', 0) > 0.5),
                'roc_auc': analysis['ppl'].get('roc_auc', 0),
                'confidence': 'high' if analysis['ppl'].get('ttest_significant', False) else 'low'
            },
            'accuracy_hypothesis': {
                'statement': 'Members have higher prediction accuracy',
                'verified': bool(analysis['accuracy']['roc_auc'] > 0.5),
                'roc_auc': analysis['accuracy']['roc_auc'],
                'confidence': 'high' if analysis['accuracy']['ttest_significant'] else 'low'
            }
        }

        self.results['analysis'] = analysis
        return analysis

    def plot_results(self):
        """Generate visualizations."""
        member_ppl = np.array(self.results['member_ppl'])
        nonmember_ppl = np.array(self.results['nonmember_ppl'])
        member_acc = np.array(self.results['member_accuracy'])
        nonmember_acc = np.array(self.results['nonmember_accuracy'])
        analysis = self.results['analysis']

        # Filter out inf values for plotting
        member_ppl_finite = member_ppl[np.isfinite(member_ppl)]
        nonmember_ppl_finite = nonmember_ppl[np.isfinite(nonmember_ppl)]

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. PPL distribution
        ax = axes[0, 0]
        ax.hist(member_ppl_finite, bins=30, alpha=0.6, label='Member', color='blue', density=True)
        ax.hist(nonmember_ppl_finite, bins=30, alpha=0.6, label='Non-member', color='red', density=True)
        ax.axvline(np.median(member_ppl_finite), color='blue', linestyle='--', linewidth=2,
                   label=f'Member median: {np.median(member_ppl_finite):.2f}')
        ax.axvline(np.median(nonmember_ppl_finite), color='red', linestyle='--', linewidth=2,
                   label=f'Non-member median: {np.median(nonmember_ppl_finite):.2f}')
        ax.set_xlabel('Perplexity (LOO)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'PPL Distribution (White-box)\nSampling:  {self.args.loo_sample_ratio:.0%} of tokens',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, np.percentile(np.concatenate([member_ppl_finite, nonmember_ppl_finite]), 95)])

        # 2. Accuracy distribution
        ax = axes[0, 1]
        ax.hist(member_acc, bins=20, alpha=0.6, label='Member', color='blue', density=True)
        ax.hist(nonmember_acc, bins=20, alpha=0.6, label='Non-member', color='red', density=True)
        ax.axvline(member_acc.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Member mean: {member_acc.mean():.3f}')
        ax.axvline(nonmember_acc.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Non-member mean: {nonmember_acc.mean():.3f}')
        ax.set_xlabel('Prediction Accuracy (LOO)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Accuracy Distribution (Black-box)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Box plot comparison
        ax = axes[0, 2]
        data_ppl = [member_ppl_finite, nonmember_ppl_finite]
        bp = ax.boxplot(data_ppl, tick_labels=['Member', 'Non-member'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('PPL Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        p_value = analysis['ppl'].get('ttest_pvalue', 1.0)
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        y_max = max(np.percentile(member_ppl_finite, 75), np.percentile(nonmember_ppl_finite, 75))
        ax.text(1.5, y_max * 1.1, f'p = {p_value:.4f} {significance}', fontsize=11, ha='center')

        # 4. Accuracy box plot
        ax = axes[1, 0]
        data_acc = [member_acc, nonmember_acc]
        bp = ax.boxplot(data_acc, tick_labels=['Member', 'Non-member'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Prediction Accuracy', fontsize=12)
        ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        p_value = analysis['accuracy']['ttest_pvalue']
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(1.5, max(member_acc.max(), nonmember_acc.max()) * 0.95,
                f'p = {p_value:.4f} {significance}', fontsize=11, ha='center')

        # 5. ROC Curves
        ax = axes[1, 1]

        # PPL ROC
        all_ppl_finite = np.concatenate([member_ppl_finite, nonmember_ppl_finite])
        ppl_labels = np.concatenate([np.ones(len(member_ppl_finite)), np.zeros(len(nonmember_ppl_finite))])
        fpr_ppl, tpr_ppl, _ = roc_curve(ppl_labels, -all_ppl_finite)

        # Accuracy ROC
        all_acc = np.concatenate([member_acc, nonmember_acc])
        acc_labels = np.concatenate([np.ones(len(member_acc)), np.zeros(len(nonmember_acc))])
        fpr_acc, tpr_acc, _ = roc_curve(acc_labels, all_acc)

        ax.plot(fpr_ppl, tpr_ppl, 'b-', linewidth=2,
                label=f'PPL (AUC={analysis["ppl"].get("roc_auc", 0):.4f})')
        ax.plot(fpr_acc, tpr_acc, 'r-', linewidth=2,
                label=f'Accuracy (AUC={analysis["accuracy"]["roc_auc"]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 6. Top-k accuracy comparison
        ax = axes[1, 2]
        k_values = [1, 5, 10]
        member_top_k = [analysis['top_k_accuracy'][f'top_{k}']['member_mean'] for k in k_values]
        nonmember_top_k = [analysis['top_k_accuracy'][f'top_{k}']['nonmember_mean'] for k in k_values]

        x = np.arange(len(k_values))
        width = 0.35

        ax.bar(x - width / 2, member_top_k, width, label='Member', color='blue', alpha=0.7)
        ax.bar(x + width / 2, nonmember_top_k, width, label='Non-member', color='red', alpha=0.7)

        ax.set_xlabel('Top-k', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Top-k Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        # Add value labels
        for i, (m, n) in enumerate(zip(member_top_k, nonmember_top_k)):
            ax.text(i - width / 2, m + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width / 2, n + 0.02, f'{n:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / 'leave_one_out_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot to:  {save_path}")
        plt.close()

    def save_results(self):
        """Save results to JSON."""
        # Convert to JSON-serializable format
        results_to_save = {
            'member_ppl': [float(x) if np.isfinite(x) else None for x in self.results['member_ppl']],
            'nonmember_ppl': [float(x) if np.isfinite(x) else None for x in self.results['nonmember_ppl']],
            'member_accuracy': [float(x) for x in self.results['member_accuracy']],
            'nonmember_accuracy': [float(x) for x in self.results['nonmember_accuracy']],
            'analysis': self.results['analysis'],
            'metadata': self.results['metadata']
        }

        output_file = self.output_dir / 'leave_one_out_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logging.info(f"Saved results to: {output_file}")

    def print_summary(self):
        """Print summary."""
        analysis = self.results['analysis']

        print("\n" + "=" * 80)
        print("LEAVE-ONE-OUT MEMBERSHIP INFERENCE SUMMARY")
        print("=" * 80)

        print(f"\nSampling Configuration:")
        print(f"  Token sample ratio: {analysis['sampling']['sample_ratio']:.0%}")
        print(f"  Avg tokens tested (member): {analysis['sampling']['avg_tokens_tested_member']:.1f}")
        print(f"  Avg tokens tested (non-member): {analysis['sampling']['avg_tokens_tested_nonmember']:.1f}")

        print("\n" + "-" * 80)
        print("1. PPL-BASED APPROACH (White-box)")
        print("-" * 80)

        ppl_analysis = analysis['ppl']
        print(f"\nPerplexity Statistics:")
        print(f"  Member median:      {ppl_analysis['member_median']:.2f}")
        print(f"  Non-member median: {ppl_analysis['nonmember_median']:.2f}")
        print(f"  Member mean:       {ppl_analysis['member_mean']:.2f} ± {ppl_analysis['member_std']:.2f}")
        print(f"  Non-member mean:   {ppl_analysis['nonmember_mean']:.2f} ± {ppl_analysis['nonmember_std']:.2f}")
        print(f"  Difference:        {ppl_analysis['diff']:.2f}")
        print(f"  P-value:           {ppl_analysis.get('ttest_pvalue', 'N/A')}")
        print(f"  Cohen's d:         {ppl_analysis.get('cohens_d', 'N/A')}")
        print(
            f"  ROC-AUC:           {ppl_analysis.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in ppl_analysis else "  ROC-AUC:            N/A")

        print("\n" + "-" * 80)
        print("2. PREDICTION-BASED APPROACH (Black-box)")
        print("-" * 80)

        acc_analysis = analysis['accuracy']
        print(f"\nTop-1 Accuracy:")
        print(f"  Member:        {acc_analysis['member_mean']:.4f} ± {acc_analysis['member_std']:.4f}")
        print(f"  Non-member:   {acc_analysis['nonmember_mean']:.4f} ± {acc_analysis['nonmember_std']:.4f}")
        print(f"  Difference:   {acc_analysis['diff']:.4f}")
        print(f"  P-value:      {acc_analysis['ttest_pvalue']:.6f}")
        print(f"  Cohen's d:    {acc_analysis['cohens_d']:.3f}")
        print(f"  ROC-AUC:      {acc_analysis['roc_auc']:.4f}")

        print(f"\nTop-k Accuracy:")
        for k in [1, 5, 10]:
            top_k = analysis['top_k_accuracy'][f'top_{k}']
            print(f"  Top-{k: 2d}: Member={top_k['member_mean']:.4f}, "
                  f"Non-member={top_k['nonmember_mean']:.4f}, "
                  f"AUC={top_k['roc_auc']:.4f}")

        print("\n" + "-" * 80)
        print("HYPOTHESIS VERIFICATION")
        print("-" * 80)

        for hyp_name, hyp_data in analysis['hypothesis'].items():
            status = "✓ VERIFIED" if hyp_data['verified'] else "✗ REJECTED"
            print(f"\n{hyp_name}:  {status}")
            print(f"  {hyp_data['statement']}")
            print(f"  ROC-AUC: {hyp_data['roc_auc']:.4f}")
            print(f"  Confidence: {hyp_data['confidence']}")

        print("\n" + "=" * 80 + "\n")

    def run(self):
        """Run complete pipeline."""
        logging.info("Starting Leave-One-Out analysis...")
        self.compute_all_samples()
        logging.info("Performing analysis...")
        self.analyze_results()
        logging.info("Generating visualizations...")
        self.plot_results()
        self.save_results()
        self.print_summary()
        logging.info(f"✓ Complete!  Results in:  {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out Membership Inference (Optimized)")

    parser.add_argument('--model_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/checkpoints/LLaDA-8B-Base-pretrained-pretraining-512-mimir-arxiv_32_5.0e-5_4epoch")
    parser.add_argument('--json_train_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/mimir-data/train/arxiv.json")
    parser.add_argument('--json_test_path', type=str,
                        default="/home1/yibiao/code/DLM-MIA/mimir-data/test/arxiv.json")

    # LOO sampling parameter
    parser.add_argument('--loo_sample_ratio', type=float, default=0.2,
                        help="Ratio of tokens to sample for LOO (0.2 = 20%% of tokens)")

    parser.add_argument('--num_samples', type=int, default=200,
                        help="Number of samples to test (0 = all)")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default="./leave_one_out_output")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    tester = LeaveOneOutTester(args)
    tester.run()


if __name__ == '__main__':
    main()