#!/usr/bin/env python
"""
Case Study V2: Single Mask Level (5%) Loss Analysis

改进方案：
- 只使用 5% 掩码率
- 直接使用 Loss（不转换为 PPL）
- 多次随机采样取平均（减少随机性）
"""

import os
import sys
import argparse
import json
import yaml
from typing import List, Dict, Tuple

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
    """加载本地 JSON Lines 数据集"""
    from datasets import Dataset as HFDataset

    all_data = []

    def load_jsonl_file(file_path, default_label):
        """加载 JSON Lines 文件"""
        if not os.path.exists(file_path):
            print(f"  ⚠️ 文件不存在: {file_path}")
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
                        text = item.get('text')
                        if text is None:
                            continue

                        label = item.get('label', default_label)
                        items.append({'text': str(text), 'label': int(label)})

                    except json.JSONDecodeError:
                        continue

                    if line_num % 1000 == 0:
                        print(f"     已加载 {line_num} 行.. .", end='\r')

            print(f"     ✅ 加载 {len(items)} 条数据" + " " * 20)

        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
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
        raise ValueError("未能加载任何数据")

    print(f"\n✅ 总计加载 {len(all_data)} 条数据")
    print(f"   - Member (label=1): {sum(1 for x in all_data if x['label'] == 1)}")
    print(f"   - Non-member (label=0): {sum(1 for x in all_data if x['label'] == 0)}")

    return HFDataset.from_list(all_data)


class LossAnalyzer:
    """Loss 分析器 - 只使用 5% 掩码"""

    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # 获取模型特定参数
        self.mask_id, self.shift_logits = get_model_nll_params(self.model)

        # 掩码配置
        self.mask_frac = config.get('mask_frac', 0.05)  # 固定 5%
        self.num_runs = config.get('num_runs', 5)  # 多次采样
        self.max_length = config.get('max_length', 512)

        print(f"[LossAnalyzer] 初始化完成")
        print(f"  - 掩码比例: {self.mask_frac:.1%}")
        print(f"  - 采样次数:  {self.num_runs}")
        print(f"  - Mask ID: {self.mask_id}, Shift Logits: {self.shift_logits}")
        print(f"  - Max Length: {self.max_length}")

    @torch.no_grad()
    def analyze_single_text(self, text: str) -> Dict:
        """
        分析单个文本

        Returns:
            {
                'losses': [loss1, loss2, ...],  # 每次采样的 loss
                'mean_loss': float,              # 平均 loss
                'std_loss': float,               # loss 标准差
                'min_loss': float,               # 最小 loss
                'max_loss': float                # 最大 loss
            }
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
            return {'losses': [], 'mean_loss': float('inf'), 'std_loss': 0,
                    'min_loss': float('inf'), 'max_loss': float('inf')}

        # 计算需要掩码的 token 数量
        num_to_mask = max(1, int(round(self.mask_frac * valid_length)))

        losses = []

        # 多次随机采样
        for run in range(self.num_runs):
            # 随机选择掩码位置
            valid_positions = torch.where(attention_mask[0])[0]

            if len(valid_positions) < num_to_mask:
                num_to_mask = len(valid_positions)

            perm = torch.randperm(len(valid_positions), device=self.device)
            chosen_indices = valid_positions[perm[: num_to_mask]]

            # 创建掩码
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            mask[0, chosen_indices] = True

            # 掩码输入
            masked_ids = input_ids.clone()
            masked_ids[mask] = self.mask_id

            # 前向传播
            out = self.model(
                input_ids=masked_ids,
                attention_mask=attention_mask if not self.shift_logits else None
            )
            logits = out.logits if hasattr(out, 'logits') else out[0]

            if self.shift_logits:
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

            # 计算交叉熵损失（只针对掩码位置）
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction='none'
            ).view(1, L).float()

            # 平均 loss
            avg_loss = ce[mask].mean().item()
            losses.append(avg_loss)

        # 统计
        losses = np.array(losses)
        return {
            'losses': losses.tolist(),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'min_loss': float(np.min(losses)),
            'max_loss': float(np.max(losses))
        }

    def analyze_dataset(self, dataset, num_samples_per_class: int = 20):
        """
        分析数据集中的样本

        Returns:
            results: Dict with member/non-member results
        """
        results = {
            'member': {'texts': [], 'losses': [], 'mean_losses': [], 'std_losses': []},
            'non_member': {'texts': [], 'losses': [], 'mean_losses': [], 'std_losses': []}
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
            result = self.analyze_single_text(text)

            if result['losses']:
                results['member']['texts'].append(text[: 100])
                results['member']['losses'].append(result['losses'])
                results['member']['mean_losses'].append(result['mean_loss'])
                results['member']['std_losses'].append(result['std_loss'])

        # 分析 non-member 样本
        print(f"分析 {len(selected_non_member)} 个 Non-member 样本...")
        for idx in tqdm(selected_non_member, desc="Non-member"):
            text = dataset[int(idx)]['text']
            result = self.analyze_single_text(text)

            if result['losses']:
                results['non_member']['texts'].append(text[:100])
                results['non_member']['losses'].append(result['losses'])
                results['non_member']['mean_losses'].append(result['mean_loss'])
                results['non_member']['std_losses'].append(result['std_loss'])

        return results


def plot_comprehensive_analysis(results: Dict, output_dir: str, config: Dict):
    """绘制综合分析图"""
    os.makedirs(output_dir, exist_ok=True)

    # 创建 2x3 子图布局
    fig = plt.figure(figsize=(18, 12))

    member_losses = np.array(results['member']['mean_losses'])
    non_member_losses = np.array(results['non_member']['mean_losses'])

    # === 子图 1: Loss 分布（直方图） ===
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(member_losses, bins=15, alpha=0.6, label='Member', color='#2E86AB', edgecolor='black')
    ax1.hist(non_member_losses, bins=15, alpha=0.6, label='Non-Member', color='#A23B72', edgecolor='black')
    ax1.axvline(np.mean(member_losses), color='#2E86AB', linestyle='--', linewidth=2,
                label=f'Member Mean: {np.mean(member_losses):.3f}')
    ax1.axvline(np.mean(non_member_losses), color='#A23B72', linestyle='--', linewidth=2,
                label=f'Non-Member Mean:  {np.mean(non_member_losses):.3f}')
    ax1.set_xlabel('Loss (5% Mask)', fontweight='bold')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Loss Distribution Comparison', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # === 子图 2: Loss 分布（箱线图） ===
    ax2 = plt.subplot(2, 3, 2)
    box_data = [member_losses, non_member_losses]
    bp = ax2.boxplot(box_data, labels=['Member', 'Non-Member'],
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    means = [np.mean(member_losses), np.mean(non_member_losses)]
    ax2.scatter([1, 2], means, color='yellow', s=100, zorder=3,
                edgecolors='black', linewidths=2, label='Mean')
    ax2.set_ylabel('Loss (5% Mask)', fontweight='bold')
    ax2.set_title('Loss Distribution (Box Plot)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # === 子图 3: 个体 Loss 对比 ===
    ax3 = plt.subplot(2, 3, 3)
    x_member = np.arange(len(member_losses))
    x_non_member = np.arange(len(non_member_losses))
    ax3.scatter(x_member, member_losses, alpha=0.6, label='Member', color='#2E86AB', s=50)
    ax3.scatter(x_non_member, non_member_losses, alpha=0.6, label='Non-Member', color='#A23B72', s=50, marker='s')
    ax3.axhline(np.mean(member_losses), color='#2E86AB', linestyle='--', linewidth=2)
    ax3.axhline(np.mean(non_member_losses), color='#A23B72', linestyle='--', linewidth=2)
    ax3.set_xlabel('Sample Index', fontweight='bold')
    ax3.set_ylabel('Loss', fontweight='bold')
    ax3.set_title('Individual Loss Values', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === 子图 4: 采样稳定性分析 ===
    ax4 = plt.subplot(2, 3, 4)
    member_stds = np.array(results['member']['std_losses'])
    non_member_stds = np.array(results['non_member']['std_losses'])
    ax4.hist(member_stds, bins=15, alpha=0.6, label='Member', color='#2E86AB', edgecolor='black')
    ax4.hist(non_member_stds, bins=15, alpha=0.6, label='Non-Member', color='#A23B72', edgecolor='black')
    ax4.set_xlabel('Loss Std Dev (across runs)', fontweight='bold')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Sampling Stability', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # === 子图 5: ROC 曲线 ===
    from sklearn.metrics import roc_curve, auc
    ax5 = plt.subplot(2, 3, 5)

    all_losses = np.concatenate([member_losses, non_member_losses])
    all_labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])

    # 注意：loss 越小越可能是 member，所以用 -loss
    fpr, tpr, _ = roc_curve(all_labels, -all_losses)
    roc_auc = auc(fpr, tpr)

    ax5.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax5.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax5.set_xlabel('False Positive Rate', fontweight='bold')
    ax5.set_ylabel('True Positive Rate', fontweight='bold')
    ax5.set_title('ROC Curve', fontweight='bold')
    ax5.legend(loc="lower right")
    ax5.grid(True, alpha=0.3)

    # === 子图 6: 统计摘要表格 ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    from scipy import stats
    t_stat, p_value = stats.ttest_ind(member_losses, non_member_losses)
    pooled_std = np.sqrt((np.std(member_losses) ** 2 + np.std(non_member_losses) ** 2) / 2)
    cohens_d = (np.mean(member_losses) - np.mean(non_member_losses)) / pooled_std if pooled_std > 0 else 0

    loss_diff = np.mean(member_losses) - np.mean(non_member_losses)
    loss_diff_pct = (loss_diff / abs(np.mean(non_member_losses))) * 100 if np.mean(non_member_losses) != 0 else 0

    stats_data = [
        ['Metric', 'Member', 'Non-Member', 'Difference'],
        ['Mean Loss', f'{np.mean(member_losses):.4f}', f'{np.mean(non_member_losses):.4f}',
         f'{loss_diff:.4f}'],
        ['Std Loss', f'{np.std(member_losses):.4f}', f'{np.std(non_member_losses):.4f}', '-'],
        ['Median Loss', f'{np.median(member_losses):.4f}', f'{np.median(non_member_losses):.4f}',
         f'{np.median(member_losses) - np.median(non_member_losses):.4f}'],
        ['', '', '', ''],
        ['AUC', f'{roc_auc:.4f}', '', ''],
        ['P-value', f'{p_value:.6f}', '', ''],
        ['Cohen\'s d', f'{cohens_d:.4f}', '', ''],
        ['Rel.  Diff %', f'{loss_diff_pct:.2f}%', '', ''],
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
    output_path = os.path.join(output_dir, 'loss_analysis_v2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 分析图已保存:  {output_path}")
    plt.close()


def save_results_json(results: Dict, output_dir: str):
    """保存结果为 JSON"""
    output_path = os.path.join(output_dir, 'loss_analysis_v2_results.json')

    json_results = {
        'member': {
            'num_samples': len(results['member']['texts']),
            'texts': results['member']['texts'],
            'mean_losses': [float(x) for x in results['member']['mean_losses']],
            'std_losses': [float(x) for x in results['member']['std_losses']],
            'overall_mean': float(np.mean(results['member']['mean_losses'])),
            'overall_std': float(np.std(results['member']['mean_losses'])),
        },
        'non_member': {
            'num_samples': len(results['non_member']['texts']),
            'texts': results['non_member']['texts'],
            'mean_losses': [float(x) for x in results['non_member']['mean_losses']],
            'std_losses': [float(x) for x in results['non_member']['std_losses']],
            'overall_mean': float(np.mean(results['non_member']['mean_losses'])),
            'overall_std': float(np.std(results['non_member']['mean_losses'])),
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果 JSON 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Loss Analysis V2 - Single Mask Level (5%)")
    parser.add_argument('-c', '--config', type=str,
                        default='attack/configs/config_fond.yaml',
                        help='配置文件路径')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='每类采样数量')
    parser.add_argument('--mask-frac', type=float, default=0.90,
                        help='掩码比例 (默认 5%)')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='每个样本的采样次数')
    parser.add_argument('--output', type=str, default='./case_study_v2_output',
                        help='输出目录')
    parser.add_argument('--base-dir', type=str, default='./',
                        help='基础目录')

    args = parser.parse_args()

    # 加载配置文件
    print("=" * 80)
    print("Loss Analysis V2 - Single Mask Level (5%)")
    print("=" * 80)
    print(f"\n📄 加载配置文件: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    global_config = config.get('global', {})

    # 设置随机种子
    seed = global_config.get('seed', 42)
    set_seed(seed)
    print(f"✅ 随机种子:  {seed}")

    # 解析模型路径
    load_from_base_dir = global_config.get('load_from_base_dir', False)
    model_path = global_config.get('target_model')
    tokenizer_path = global_config.get('tokenizer', model_path)

    if not model_path:
        raise ValueError("配置文件中必须指定 target_model")

    model_path = resolve_path(model_path, args.base_dir, load_from_base_dir)
    tokenizer_path = resolve_path(tokenizer_path, args.base_dir, load_from_base_dir)

    # 加载模型
    print(f"\n📥 加载模型: {model_path}")
    ModelManager.register_custom_models()

    device_str = global_config.get('device', 'cuda')
    device = torch.device(device_str)

    model, tokenizer, device = ModelManager.init_model(
        model_path, tokenizer_path, device
    )
    print("✅ 模型加载完成")

    # 加载数据集
    datasets_config = global_config.get('datasets', [])
    if not datasets_config:
        raise ValueError("配置文件中必须指定 datasets")

    ds_config = datasets_config[0]

    print(f"\n📥 加载数据集...")

    if 'json_train_path' in ds_config and 'json_test_path' in ds_config:
        train_path = resolve_path(ds_config['json_train_path'], args.base_dir, load_from_base_dir)
        test_path = resolve_path(ds_config['json_test_path'], args.base_dir, load_from_base_dir)
        dataset = load_local_json_dataset(train_path, test_path)
        print(f"✅ 本地数据集加载完成 (总样本数: {len(dataset)})")
    else:
        ds_name = ds_config.get('name') or ds_config.get('path')
        ds_split = ds_config.get('split', 'train')
        dataset = hf_load_dataset(ds_name, split=ds_split)
        print(f"✅ HuggingFace 数据集加载完成:  {ds_name} (总样本数: {len(dataset)})")

    # 初始化分析器
    analysis_config = {
        'mask_frac': args.mask_frac,
        'num_runs': args.num_runs,
        'max_length': global_config.get('max_length', 512),
    }

    analyzer = LossAnalyzer(model, tokenizer, device, analysis_config)

    # 运行分析
    print(f"\n🔬 开始分析 (每类 {args.num_samples} 个样本)...")
    results = analyzer.analyze_dataset(dataset, num_samples_per_class=args.num_samples)

    # 打印统计信息
    print("\n" + "=" * 80)
    print("📊 统计摘要")
    print("=" * 80)

    member_losses = np.array(results['member']['mean_losses'])
    non_member_losses = np.array(results['non_member']['mean_losses'])

    print(f"\n✅ MEMBER 样本:")
    print(f"  - 样本数: {len(member_losses)}")
    print(f"  - 平均 Loss: {np.mean(member_losses):.6f}")
    print(f"  - 标准差: {np.std(member_losses):.6f}")
    print(f"  - 中位数: {np.median(member_losses):.6f}")
    print(f"  - 范围: [{np.min(member_losses):.6f}, {np.max(member_losses):.6f}]")

    print(f"\n❌ NON-MEMBER 样本:")
    print(f"  - 样本数: {len(non_member_losses)}")
    print(f"  - 平均 Loss: {np.mean(non_member_losses):.6f}")
    print(f"  - 标准差: {np.std(non_member_losses):.6f}")
    print(f"  - 中位数: {np.median(non_member_losses):.6f}")
    print(f"  - 范围:  [{np.min(non_member_losses):.6f}, {np.max(non_member_losses):.6f}]")

    # 对比分析
    loss_diff = np.mean(member_losses) - np.mean(non_member_losses)
    loss_diff_pct = (loss_diff / abs(np.mean(non_member_losses))) * 100 if np.mean(non_member_losses) != 0 else 0

    print(f"\n🔍 对比分析:")
    print(f"  - Loss 差异 (M - NM): {loss_diff:.6f}")
    print(f"  - 相对差异:  {loss_diff_pct:.2f}%")

    if loss_diff < 0:
        print(f"  ✅ Member Loss 更低 (符合预期)")
    else:
        print(f"  ⚠️ Member Loss 更高 (不符合预期)")

    # 统计显著性
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    t_stat, p_value = stats.ttest_ind(member_losses, non_member_losses)
    pooled_std = np.sqrt((np.std(member_losses) ** 2 + np.std(non_member_losses) ** 2) / 2)
    cohens_d = (np.mean(member_losses) - np.mean(non_member_losses)) / pooled_std if pooled_std > 0 else 0

    all_losses = np.concatenate([member_losses, non_member_losses])
    all_labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    auc_score = roc_auc_score(all_labels, -all_losses)

    print(f"\n📈 统计检验:")
    print(f"  - T 统计量: {t_stat:.4f}")
    print(f"  - P 值: {p_value:.6f} {'(显著)' if p_value < 0.05 else '(不显著)'}")
    print(f"  - Cohen's d: {cohens_d:.4f}")
    print(f"  - AUC: {auc_score:.4f}")

    if auc_score > 0.7:
        print(f"  ✅ 分类能力:  较好 (AUC > 0.7)")
    elif auc_score > 0.6:
        print(f"  ⚠️ 分类能力: 一般 (0.6 < AUC ≤ 0.7)")
    else:
        print(f"  ❌ 分类能力:  较弱 (AUC ≤ 0.6)")

    # 绘图和保存
    print(f"\n🎨 生成可视化...")
    plot_comprehensive_analysis(results, args.output, analysis_config)
    save_results_json(results, args.output)

    # 快速摘要
    print(f"\n{'#' * 80}")
    print(f"# 📋 快速摘要 (V2 - 单一掩码级别)")
    print(f"{'#' * 80}")
    print(f"""
配置: 
- 掩码比例: {args.mask_frac:.1%}
- 采样次数: {args.num_runs}
- 每类样本数: {args.num_samples}

结果:
- Member Loss:       {np.mean(member_losses):.6f} ± {np.std(member_losses):.6f}
- Non-Member Loss:   {np.mean(non_member_losses):.6f} ± {np.std(non_member_losses):.6f}
- 差异:             {loss_diff:.6f} ({loss_diff_pct:.2f}%)
- AUC:             {auc_score:.4f}
- P值:             {p_value:.6f} {'(显著)' if p_value < 0.05 else '(不显著)'}
- Cohen's d:       {cohens_d:.4f}

结论:  {'✅ 有区分性' if auc_score > 0.6 and p_value < 0.05 else '❌ 区分性不足'}
    """)
    print(f"{'#' * 80}\n")

    print("\n" + "=" * 80)
    print(f"✅ 分析完成!  所有结果已保存到:  {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()