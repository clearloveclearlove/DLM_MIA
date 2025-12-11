#!/usr/bin/env python
"""
掩码比例扫描分析 - 分析不同掩码比例下的 AUROC 变化趋势
用法:  CUDA_VISIBLE_DEVICES=0 python case_study_mask_sweep.py
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy import stats

sns.set_style("whitegrid")
sys.path.insert(0, os.path.dirname(__file__))

from attack.misc.models import ModelManager
from attack.misc.utils import set_seed, resolve_path
from attack.attacks.utils import get_model_nll_params


def load_local_json_dataset(train_path: str, test_path: str) -> Dataset:
    """加载本地数据集"""
    all_data = []

    for file_path, label in [(train_path, 1), (test_path, 0)]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('text'):
                    all_data.append({'text': item['text'], 'label': label})

    print(f"✅ 加载 {len(all_data)} 条数据")
    return Dataset.from_list(all_data)


class MaskAnalyzer:
    def __init__(self, model, tokenizer, device, max_length=512, num_runs=5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_runs = num_runs
        self.mask_id, self.shift_logits = get_model_nll_params(model)

    @torch.no_grad()
    def get_loss(self, text: str, mask_frac: float) -> float:
        """计算指定掩码比例下的平均loss"""
        encoded = self.tokenizer.encode_plus(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        valid_length = int(attention_mask.sum().item())

        if valid_length == 0:
            return float('inf')

        num_to_mask = max(1, int(round(mask_frac * valid_length)))
        losses = []

        for _ in range(self.num_runs):
            valid_pos = torch.where(attention_mask[0])[0]
            perm = torch.randperm(len(valid_pos), device=self.device)
            chosen = valid_pos[perm[:num_to_mask]]

            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            mask[0, chosen] = True

            masked_ids = input_ids.clone()
            masked_ids[mask] = self.mask_id

            out = self.model(input_ids=masked_ids,
                             attention_mask=attention_mask if not self.shift_logits else None)
            logits = out.logits if hasattr(out, 'logits') else out[0]

            if self.shift_logits:
                logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

            ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 input_ids.view(-1), reduction='none').view(1, -1).float()

            losses.append(ce[mask].mean().item())

        return float(np.mean(losses))


def analyze_mask_ratio(analyzer, dataset, mask_frac, num_samples=100):
    """分析单个掩码比例"""
    # 采样
    member_idx = [i for i, x in enumerate(dataset) if x['label'] == 1]
    non_member_idx = [i for i, x in enumerate(dataset) if x['label'] == 0]

    np.random.seed(42)
    sel_member = np.random.choice(member_idx, min(num_samples, len(member_idx)), replace=False)
    sel_non = np.random.choice(non_member_idx, min(num_samples, len(non_member_idx)), replace=False)

    # 计算loss
    member_losses = []
    for idx in tqdm(sel_member, desc=f"Member ({mask_frac:.0%})"):
        loss = analyzer.get_loss(dataset[int(idx)]['text'], mask_frac)
        if loss != float('inf'):
            member_losses.append(loss)

    non_losses = []
    for idx in tqdm(sel_non, desc=f"Non-Member ({mask_frac:.0%})"):
        loss = analyzer.get_loss(dataset[int(idx)]['text'], mask_frac)
        if loss != float('inf'):
            non_losses.append(loss)

    member_losses = np.array(member_losses)
    non_losses = np.array(non_losses)

    # 统计
    all_losses = np.concatenate([member_losses, non_losses])
    all_labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_losses))])
    auroc = roc_auc_score(all_labels, -all_losses)

    t_stat, p_value = stats.ttest_ind(member_losses, non_losses)
    pooled_std = np.sqrt((np.std(member_losses) ** 2 + np.std(non_losses) ** 2) / 2)
    cohens_d = (np.mean(member_losses) - np.mean(non_losses)) / pooled_std if pooled_std > 0 else 0

    return {
        'auroc': auroc,
        'member_mean': np.mean(member_losses),
        'non_member_mean': np.mean(non_losses),
        'cohens_d': cohens_d,
        'p_value': p_value
    }


def plot_results(results, output_dir):
    """绘制结果"""
    os.makedirs(output_dir, exist_ok=True)

    mask_fracs = sorted(results.keys())
    aurocs = [results[mf]['auroc'] for mf in mask_fracs]
    cohens_ds = [results[mf]['cohens_d'] for mf in mask_fracs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUROC趋势
    ax1.plot(mask_fracs, aurocs, marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax1.set_xlabel('Mask Ratio', fontweight='bold', fontsize=12)
    ax1.set_ylabel('AUROC', fontweight='bold', fontsize=12)
    ax1.set_title('AUROC vs Mask Ratio', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    max_idx = np.argmax(aurocs)
    ax1.annotate(f'Best: {aurocs[max_idx]:.4f}\n@{mask_fracs[max_idx]:.0%}',
                 xy=(mask_fracs[max_idx], aurocs[max_idx]),
                 xytext=(10, -30), textcoords='offset points',
                 fontsize=10, color='red', weight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Cohen's d趋势
    ax2.plot(mask_fracs, cohens_ds, marker='s', linewidth=2.5, markersize=10, color='#A23B72')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_xlabel('Mask Ratio', fontweight='bold', fontsize=12)
    ax2.set_ylabel("Cohen's d", fontweight='bold', fontsize=12)
    ax2.set_title("Effect Size vs Mask Ratio", fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/auroc_trend.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表保存到:  {output_dir}/auroc_trend.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='attack/configs/config_fond.yaml')
    parser.add_argument('--mask-fracs', type=float, nargs='+',
                        default=[0.01, 0.05, 0.10, 0.20, 0.50, 0.90])
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--output', default='./mask_sweep_output')
    args = parser.parse_args()

    print("=" * 80)
    print("掩码比例扫描分析 - AUROC 趋势")
    print("=" * 80)

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    global_config = config['global']

    set_seed(global_config.get('seed', 42))

    # 加载模型
    model_path = global_config['target_model']
    print(f"\n📥 加载模型:  {model_path}")
    ModelManager.register_custom_models()

    device = torch.device(global_config.get('device', 'cuda'))
    model, tokenizer, device = ModelManager.init_model(model_path, model_path, device)
    print("✅ 模型加载完成")

    # 加载数据
    ds_config = global_config['datasets'][0]
    train_path = ds_config['json_train_path']
    test_path = ds_config['json_test_path']
    dataset = load_local_json_dataset(train_path, test_path)

    # 分析
    analyzer = MaskAnalyzer(model, tokenizer, device,
                            global_config.get('max_length', 512), args.num_runs)

    results = {}
    print(f"\n🔬 测试掩码比例:  {[f'{x:.0%}' for x in args.mask_fracs]}")
    print(f"每类样本数: {args.num_samples}, 采样次数: {args.num_runs}\n")

    for mask_frac in args.mask_fracs:
        print(f"\n{'=' * 80}")
        print(f"分析掩码比例:  {mask_frac:.0%}")
        print('=' * 80)

        result = analyze_mask_ratio(analyzer, dataset, mask_frac, args.num_samples)
        results[mask_frac] = result

        print(f"✅ AUROC: {result['auroc']:.4f}, Cohen's d: {result['cohens_d']:.4f}, "
              f"P-value: {result['p_value']:.6f}")

    # 汇总
    print("\n" + "=" * 80)
    print("📊 汇总结果")
    print("=" * 80)
    for mf in sorted(results.keys()):
        r = results[mf]
        print(f"{mf: 5.0%}: AUROC={r['auroc']:.4f}, Cohen's d={r['cohens_d']: 6.3f}, P={r['p_value']:.6f}")

    best_mf = max(results.keys(), key=lambda x: results[x]['auroc'])
    print(f"\n🏆 最优掩码比例: {best_mf:.0%} (AUROC: {results[best_mf]['auroc']:.4f})")

    # 保存
    plot_results(results, args.output)

    with open(f'{args.output}/results.json', 'w') as f:
        json.dump({f'{k:.2%}': v for k, v in results.items()}, f, indent=2)
    print(f"✅ 结果保存到: {args.output}/results.json")

    print("\n" + "=" * 80)
    print("✅ 分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()