#!/usr/bin/env python
"""
Case Study: Progressive Masking PPL Trends Analysis

分析 member 和 non-member 样本在渐进式掩码下的 PPL 变化趋势
基于 config_fond.yaml 的配置风格
"""

import os
import sys
import argparse
import json
import yaml
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset as hf_load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from attack.misc.models import ModelManager
from attack.misc.utils import set_seed, resolve_path
from attack.attacks.utils import get_model_nll_params


def load_local_json_dataset(train_path: str, test_path: str) -> Dataset:
    """
    加载本地 JSON Lines 数据集

    期望格式（JSON Lines）：
    {"text": ".. .", ... }
    {"text": "...", ...}
    """
    from datasets import Dataset as HFDataset

    all_data = []

    def load_jsonl_file(file_path, default_label):
        """加载 JSON Lines 文件"""
        if not os.path.exists(file_path):
            print(f"  ⚠️ 文件不存在:  {file_path}")
            return []

        print(f"  📂 加载:  {file_path}")
        items = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)

                        # 提取文本
                        text = item.get('text')
                        if text is None:
                            print(f"     ⚠️ 第 {line_num} 行缺少 'text' 字段")
                            continue

                        # 提取或设置标签
                        label = item.get('label', default_label)

                        items.append({
                            'text': str(text),
                            'label': int(label)
                        })

                    except json.JSONDecodeError as e:
                        print(f"     ⚠️ 第 {line_num} 行 JSON 解析失败: {e}")
                        continue

                    # 显示进度
                    if line_num % 1000 == 0:
                        print(f"     已加载 {line_num} 行.. .", end='\r')

            print(f"     ✅ 加载 {len(items)} 条数据" + " " * 20)

        except Exception as e:
            print(f"  ❌ 加载失败:  {e}")
            return []

        return items

    # 加载训练集（member, label=1）
    print("\n加载训练集 (Member):")
    train_items = load_jsonl_file(train_path, default_label=1)
    all_data.extend(train_items)

    # 加载测试集（non-member, label=0）
    print("\n加载测试集 (Non-member):")
    test_items = load_jsonl_file(test_path, default_label=0)
    all_data.extend(test_items)

    if not all_data:
        raise ValueError(f"""
未能加载任何数据。请检查：
1. 文件是否存在: 
   - {train_path}
   - {test_path}
2. JSON Lines 格式是否正确（每行一个 JSON 对象）
3. 每行是否包含 'text' 字段
""")

    print(f"\n✅ 总计加载 {len(all_data)} 条数据")
    print(f"   - Member (label=1): {sum(1 for x in all_data if x['label'] == 1)}")
    print(f"   - Non-member (label=0): {sum(1 for x in all_data if x['label'] == 0)}")

    return HFDataset.from_list(all_data)


class PPLTrendAnalyzer:
    """PPL 趋势分析器"""

    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # 获取模型特定参数
        self.mask_id, self.shift_logits = get_model_nll_params(self.model)

        # 掩码配置
        self.steps = config.get('steps', 10)
        self.min_mask_frac = config.get('min_mask_frac', 0.05)
        self.max_mask_frac = config.get('max_mask_frac', 0.90)
        self.mask_schedule = config.get('mask_schedule', 'linear')
        self.max_length = config.get('max_length', 512)

        print(f"[PPLTrendAnalyzer] 初始化完成")
        print(f"  - 步数: {self.steps}")
        print(f"  - 掩码范围: [{self.min_mask_frac:.2f}, {self.max_mask_frac:.2f}]")  # ← 修复这里
        print(f"  - 调度策略: {self.mask_schedule}")
        print(f"  - Mask ID: {self.mask_id}, Shift Logits: {self.shift_logits}")
        print(f"  - Max Length: {self.max_length}")

    @torch.no_grad()
    def analyze_single_text(self, text: str) -> Tuple[List[float], List[float]]:
        """
        分析单个文本的 PPL 趋势

        Returns:
            (mask_fractions, ppl_values)
        """
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]  # (1, L)
        attention_mask = encoded["attention_mask"].bool()  # (1, L)
        L = input_ids.size(1)

        # 有效长度
        valid_length = int(attention_mask.sum().item())
        if valid_length == 0:
            return [], []

        mask_fractions = []
        ppl_values = []
        cumulative_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # 渐进式掩码
        for step in range(self.steps):
            # 计算掩码比例
            frac = self._compute_mask_fraction(step)
            mask_fractions.append(frac)

            # 计算应掩码的总数
            desired_total = max(1, int(round(frac * valid_length)))
            current_total = int((cumulative_mask[0] & attention_mask[0]).sum().item())
            to_add = max(0, desired_total - current_total)

            # 添加新掩码
            if to_add > 0:
                unmasked_valid = (~cumulative_mask[0]) & attention_mask[0]
                candidates = torch.where(unmasked_valid)[0]

                if candidates.numel() > 0:
                    to_add = min(to_add, candidates.numel())
                    perm = torch.randperm(candidates.numel(), device=self.device)
                    chosen = candidates[perm[:to_add]]
                    cumulative_mask[0, chosen] = True

            # 创建掩码输入
            masked_ids = input_ids.clone()
            masked_ids[cumulative_mask] = self.mask_id

            # 前向传播
            out = self.model(
                input_ids=masked_ids,
                attention_mask=attention_mask if not self.shift_logits else None
            )
            logits = out.logits if hasattr(out, 'logits') else out[0]

            if self.shift_logits:
                logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

            # 计算损失（只针对掩码位置）
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction='none'
            ).view(1, L).float()

            # 计算 PPL
            masked_positions = cumulative_mask[0] & attention_mask[0]
            if masked_positions.sum() > 0:
                avg_loss = ce[0, masked_positions].mean().item()
                ppl = np.exp(avg_loss)
            else:
                ppl = float('inf')

            ppl_values.append(ppl)

        return mask_fractions, ppl_values

    def _compute_mask_fraction(self, step: int) -> float:
        """计算掩码比例"""
        if self.mask_schedule == "geometric":
            if self.steps == 1:
                return self.max_mask_frac
            ratio = (self.max_mask_frac / max(self.min_mask_frac, 1e-6)) ** (1 / (self.steps - 1))
            frac = self.min_mask_frac * (ratio ** step)
        else:
            # 线性
            if self.steps == 1:
                frac = self.max_mask_frac
            else:
                frac = self.min_mask_frac + (self.max_mask_frac - self.min_mask_frac) * step / (self.steps - 1)

        return min(max(frac, 0.0), 1.0)

    def analyze_dataset(self, dataset, num_samples_per_class: int = 10):
        """
        分析数据集中的样本

        Args:
            dataset: 包含 'text' 和 'label' 的数据集
            num_samples_per_class: 每类采样的样本数

        Returns:
            results: Dict with member/non-member results
        """
        results = {
            'member': {'texts': [], 'mask_fractions': [], 'ppl_sequences': [], 'slopes': []},
            'non_member': {'texts': [], 'mask_fractions': [], 'ppl_sequences': [], 'slopes': []}
        }

        # 分离 member 和 non-member
        member_indices = [i for i, label in enumerate(dataset['label']) if label == 1]
        non_member_indices = [i for i, label in enumerate(dataset['label']) if label == 0]

        print(f"\n数据集统计:")
        print(f"  - Member 样本: {len(member_indices)}")
        print(f"  - Non-member 样本: {len(non_member_indices)}")

        # 随机采样
        np.random.seed(42)
        selected_member = np.random.choice(member_indices,
                                           min(num_samples_per_class, len(member_indices)),
                                           replace=False)
        selected_non_member = np.random.choice(non_member_indices,
                                               min(num_samples_per_class, len(non_member_indices)),
                                               replace=False)

        # 分析 member 样本
        print(f"\n分析 {len(selected_member)} 个 Member 样本...")
        for idx in tqdm(selected_member, desc="Member"):
            text = dataset[int(idx)]['text']
            mask_fracs, ppl_vals = self.analyze_single_text(text)

            if len(ppl_vals) > 0:
                results['member']['texts'].append(text[: 100])
                results['member']['mask_fractions'].append(mask_fracs)
                results['member']['ppl_sequences'].append(ppl_vals)

                # 计算斜率
                slope = self._compute_slope(np.arange(len(ppl_vals)), np.array(ppl_vals))
                results['member']['slopes'].append(slope)

        # 分析 non-member 样本
        print(f"分析 {len(selected_non_member)} 个 Non-member 样本...")
        for idx in tqdm(selected_non_member, desc="Non-member"):
            text = dataset[int(idx)]['text']
            mask_fracs, ppl_vals = self.analyze_single_text(text)

            if len(ppl_vals) > 0:
                results['non_member']['texts'].append(text[:100])
                results['non_member']['mask_fractions'].append(mask_fracs)
                results['non_member']['ppl_sequences'].append(ppl_vals)

                # 计算斜率
                slope = self._compute_slope(np.arange(len(ppl_vals)), np.array(ppl_vals))
                results['non_member']['slopes'].append(slope)

        return results

    def _compute_slope(self, x, y):
        """线性回归计算斜率"""
        if len(x) < 2:
            return 0.0

        x_mean = x.mean()
        y_mean = y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator


def plot_comprehensive_analysis(results: Dict, output_dir: str):
    """绘制综合分析图"""
    os.makedirs(output_dir, exist_ok=True)

    # 创建 2x3 子图布局
    fig = plt.figure(figsize=(18, 12))

    # === 子图 1: 个体 PPL 曲线（Member） ===
    ax1 = plt.subplot(2, 3, 1)
    for i, (mask_fracs, ppl_vals) in enumerate(zip(results['member']['mask_fractions'][:5],
                                                   results['member']['ppl_sequences'][:5])):
        ax1.plot(mask_fracs, ppl_vals, marker='o', alpha=0.6, label=f'Sample {i + 1}')
    ax1.set_xlabel('Mask Fraction')
    ax1.set_ylabel('Perplexity (PPL)')
    ax1.set_title('Member Samples - Individual PPL Curves (Top 5)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # === 子图 2: 个体 PPL 曲线（Non-member） ===
    ax2 = plt.subplot(2, 3, 2)
    for i, (mask_fracs, ppl_vals) in enumerate(zip(results['non_member']['mask_fractions'][:5],
                                                   results['non_member']['ppl_sequences'][:5])):
        ax2.plot(mask_fracs, ppl_vals, marker='s', alpha=0.6, label=f'Sample {i + 1}')
    ax2.set_xlabel('Mask Fraction')
    ax2.set_ylabel('Perplexity (PPL)')
    ax2.set_title('Non-Member Samples - Individual PPL Curves (Top 5)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # === 子图 3: 平均 PPL 对比（带误差带） ===
    ax3 = plt.subplot(2, 3, 3)

    # Member 平均曲线
    member_ppls = np.array(results['member']['ppl_sequences'])
    member_mean = np.mean(member_ppls, axis=0)
    member_std = np.std(member_ppls, axis=0)
    mask_fracs = results['member']['mask_fractions'][0]

    ax3.plot(mask_fracs, member_mean, marker='o', linewidth=2.5,
             label='Member (avg)', color='#2E86AB')
    ax3.fill_between(mask_fracs,
                     member_mean - member_std,
                     member_mean + member_std,
                     alpha=0.2, color='#2E86AB')

    # Non-member 平均曲线
    non_member_ppls = np.array(results['non_member']['ppl_sequences'])
    non_member_mean = np.mean(non_member_ppls, axis=0)
    non_member_std = np.std(non_member_ppls, axis=0)

    ax3.plot(mask_fracs, non_member_mean, marker='s', linewidth=2.5,
             label='Non-Member (avg)', color='#A23B72')
    ax3.fill_between(mask_fracs,
                     non_member_mean - non_member_std,
                     non_member_mean + non_member_std,
                     alpha=0.2, color='#A23B72')

    ax3.set_xlabel('Mask Fraction', fontweight='bold')
    ax3.set_ylabel('Perplexity (PPL)', fontweight='bold')
    ax3.set_title('Average PPL Comparison (Mean ± Std)', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # === 子图 4: 斜率分布对比 (直方图) ===
    ax4 = plt.subplot(2, 3, 4)

    member_slopes = results['member']['slopes']
    non_member_slopes = results['non_member']['slopes']

    ax4.hist(member_slopes, bins=15, alpha=0.6, label='Member', color='#2E86AB', edgecolor='black')
    ax4.hist(non_member_slopes, bins=15, alpha=0.6, label='Non-Member', color='#A23B72', edgecolor='black')
    ax4.axvline(np.mean(member_slopes), color='#2E86AB', linestyle='--', linewidth=2,
                label=f'Member Mean: {np.mean(member_slopes):.2f}')
    ax4.axvline(np.mean(non_member_slopes), color='#A23B72', linestyle='--', linewidth=2,
                label=f'Non-Member Mean:  {np.mean(non_member_slopes):.2f}')
    ax4.set_xlabel('Slope (PPL Growth Rate)', fontweight='bold')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Slope Distribution Comparison', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # === 子图 5: 斜率分布对比 (箱线图) ===
    ax5 = plt.subplot(2, 3, 5)

    box_data = [member_slopes, non_member_slopes]
    bp = ax5.boxplot(box_data, labels=['Member', 'Non-Member'],
                     patch_artist=True, widths=0.6)

    # 设置颜色
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')

    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    ax5.set_ylabel('Slope (PPL Growth Rate)', fontweight='bold')
    ax5.set_title('Slope Distribution (Box Plot)', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 添加均值点
    means = [np.mean(member_slopes), np.mean(non_member_slopes)]
    ax5.scatter([1, 2], means, color='yellow', s=100, zorder=3,
                edgecolors='black', linewidths=2, label='Mean')
    ax5.legend()

    # === 子图 6: 统计摘要表格 ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # 计算统计量
    stats_data = [
        ['Metric', 'Member', 'Non-Member', 'Difference'],
        ['Mean Slope', f'{np.mean(member_slopes):.4f}', f'{np.mean(non_member_slopes):.4f}',
         f'{np.mean(member_slopes) - np.mean(non_member_slopes):.4f}'],
        ['Std Slope', f'{np.std(member_slopes):.4f}', f'{np.std(non_member_slopes):.4f}', '-'],
        ['Median Slope', f'{np.median(member_slopes):.4f}', f'{np.median(non_member_slopes):.4f}',
         f'{np.median(member_slopes) - np.median(non_member_slopes):.4f}'],
        ['', '', '', ''],
        ['Mean PPL@10%', f'{member_mean[0]:.2f}', f'{non_member_mean[0]:.2f}',
         f'{member_mean[0] - non_member_mean[0]:.2f}'],
        ['Mean PPL@50%', f'{member_mean[len(member_mean) // 2]:.2f}',
         f'{non_member_mean[len(non_member_mean) // 2]:.2f}',
         f'{member_mean[len(member_mean) // 2] - non_member_mean[len(non_member_mean) // 2]:.2f}'],
        ['Mean PPL@90%', f'{member_mean[-1]:.2f}', f'{non_member_mean[-1]:.2f}',
         f'{member_mean[-1] - non_member_mean[-1]:.2f}'],
    ]

    table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold')

    ax6.set_title('Statistical Summary', fontweight='bold', fontsize=13, pad=20)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'ppl_slope_comprehensive_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 综合分析图已保存:  {output_path}")

    plt.close()

    # === 额外绘制：PPL 增长率热力图 ===
    plot_growth_rate_heatmap(results, output_dir)


def plot_growth_rate_heatmap(results: Dict, output_dir: str):
    """绘制 PPL 增长率热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Member 热力图
    member_ppls = np.array(results['member']['ppl_sequences'])
    member_growth = np.diff(member_ppls, axis=1)  # 计算增长率

    im1 = axes[0].imshow(member_growth, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel('Mask Step Transition', fontweight='bold')
    axes[0].set_ylabel('Sample Index')
    axes[0].set_title('Member Samples - PPL Growth Rate Heatmap', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='PPL Increase')

    # Non-member 热力图
    non_member_ppls = np.array(results['non_member']['ppl_sequences'])
    non_member_growth = np.diff(non_member_ppls, axis=1)

    im2 = axes[1].imshow(non_member_growth, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_xlabel('Mask Step Transition', fontweight='bold')
    axes[1].set_ylabel('Sample Index')
    axes[1].set_title('Non-Member Samples - PPL Growth Rate Heatmap', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='PPL Increase')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ppl_growth_rate_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ PPL 增长率热力图已保存: {output_path}")
    plt.close()


def save_results_json(results: Dict, output_dir: str):
    """保存结果为 JSON"""
    output_path = os.path.join(output_dir, 'case_study_results.json')

    # 转换为可序列化格式
    json_results = {
        'member': {
            'num_samples': len(results['member']['texts']),
            'texts': results['member']['texts'],
            'ppl_sequences': [list(map(float, seq)) for seq in results['member']['ppl_sequences']],
            'slopes': [float(s) for s in results['member']['slopes']],
            'mean_slope': float(np.mean(results['member']['slopes'])),
            'std_slope': float(np.std(results['member']['slopes'])),
        },
        'non_member': {
            'num_samples': len(results['non_member']['texts']),
            'texts': results['non_member']['texts'],
            'ppl_sequences': [list(map(float, seq)) for seq in results['non_member']['ppl_sequences']],
            'slopes': [float(s) for s in results['non_member']['slopes']],
            'mean_slope': float(np.mean(results['non_member']['slopes'])),
            'std_slope': float(np.std(results['non_member']['slopes'])),
        },
        'mask_fractions': results['member']['mask_fractions'][0] if results['member']['mask_fractions'] else []
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果 JSON 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PPL Slope Case Study - 基于 config_fond.yaml 风格")
    parser.add_argument('-c', '--config', type=str,
                        default='attack/configs/config_fond.yaml',
                        help='配置文件路径')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='每类采样数量（覆盖配置文件）')
    parser.add_argument('--steps', type=int, default=None,
                        help='掩码步数（覆盖配置文件）')
    parser.add_argument('--output', type=str, default='./case_study_output',
                        help='输出目录')
    parser.add_argument('--base-dir', type=str, default='./',
                        help='基础目录（用于解析相对路径）')

    args = parser.parse_args()

    # === 加载配置文件 ===
    print("=" * 80)
    print("PPL Slope Case Study - Progressive Masking Analysis")
    print("=" * 80)
    print(f"\n📄 加载配置文件: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    global_config = config.get('global', {})

    # 设置随机种子
    seed = global_config.get('seed', 42)
    set_seed(seed)
    print(f"✅ 随机种子:  {seed}")

    # === 解析模型路径 ===
    load_from_base_dir = global_config.get('load_from_base_dir', False)
    model_path = global_config.get('target_model')
    tokenizer_path = global_config.get('tokenizer', model_path)

    if not model_path:
        raise ValueError("配置文件中必须指定 target_model")

    # 解析路径
    model_path = resolve_path(model_path, args.base_dir, load_from_base_dir)
    tokenizer_path = resolve_path(tokenizer_path, args.base_dir, load_from_base_dir)

    # === 加载模型 ===
    print(f"\n📥 加载模型: {model_path}")
    ModelManager.register_custom_models()

    device_str = global_config.get('device', 'cuda')
    device = torch.device(device_str)

    model, tokenizer, device = ModelManager.init_model(
        model_path, tokenizer_path, device
    )
    print("✅ 模型加载完成")

    # === 加载数据集 ===
    datasets_config = global_config.get('datasets', [])
    if not datasets_config:
        raise ValueError("配置文件中必须指定 datasets")

    ds_config = datasets_config[0]  # 使用第一个数据集

    print(f"\n📥 加载数据集...")

    # 检查是本地 JSON 还是 HuggingFace 数据集
    if 'json_train_path' in ds_config and 'json_test_path' in ds_config:
        # 本地 JSON 格式
        train_path = resolve_path(ds_config['json_train_path'], args.base_dir, load_from_base_dir)
        test_path = resolve_path(ds_config['json_test_path'], args.base_dir, load_from_base_dir)

        dataset = load_local_json_dataset(train_path, test_path)
        print(f"✅ 本地数据集加载完成 (总样本数: {len(dataset)})")
    else:
        # HuggingFace 数据集
        ds_name = ds_config.get('name') or ds_config.get('path')
        ds_split = ds_config.get('split', 'train')
        dataset = hf_load_dataset(ds_name, split=ds_split)
        print(f"✅ HuggingFace 数据集加载完成:  {ds_name} (总样本数: {len(dataset)})")

    # === 初始化分析器 ===
    analysis_config = {
        'steps': args.steps if args.steps else global_config.get('steps', 10),
        'min_mask_frac': 0.05,
        'max_mask_frac': 0.90,
        'mask_schedule': 'linear',
        'max_length': global_config.get('max_length', 512),
    }

    analyzer = PPLTrendAnalyzer(model, tokenizer, device, analysis_config)

    # === 运行分析 ===
    print(f"\n🔬 开始分析 (每类 {args.num_samples} 个样本)...")
    results = analyzer.analyze_dataset(dataset, num_samples_per_class=args.num_samples)

    # === 打印统计信息 ===
    print("\n" + "=" * 80)
    print("📊 统计摘要")
    print("=" * 80)
    print(f"\nMember 样本:")
    print(f"  - 分析数量: {len(results['member']['slopes'])}")
    print(f"  - 平均斜率: {np.mean(results['member']['slopes']):.6f}")
    print(f"  - 斜率标准差: {np.std(results['member']['slopes']):.6f}")
    print(f"  - 斜率中位数: {np.median(results['member']['slopes']):.6f}")

    print(f"\nNon-Member 样本:")
    print(f"  - 分析数量: {len(results['non_member']['slopes'])}")
    print(f"  - 平均斜率: {np.mean(results['non_member']['slopes']):.6f}")
    print(f"  - 斜率标准差: {np.std(results['non_member']['slopes']):.6f}")
    print(f"  - 斜率中位数: {np.median(results['non_member']['slopes']):.6f}")

    slope_diff = np.mean(results['member']['slopes']) - np.mean(results['non_member']['slopes'])
    print(f"\n📈 斜率差异 (Member - Non-Member): {slope_diff:.6f}")

    if slope_diff < 0:
        print("   ✅ Member 样本 PPL 增长更慢 (符合预期)")
    else:
        print("   ⚠️ Member 样本 PPL 增长更快 (不符合预期)")

    # === 绘图和保存 ===
    print(f"\n🎨 生成可视化...")
    plot_comprehensive_analysis(results, args.output)
    save_results_json(results, args.output)

    print("\n" + "=" * 80)
    print(f"✅ Case Study 完成!  所有结果已保存到:  {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()