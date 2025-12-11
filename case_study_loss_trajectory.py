#!/usr/bin/env python
"""
Loss Trajectory Analysis - 基于 PPL 轨迹形状的成员推断
分析不同掩码比例（时间步 t）下的 PPL 曲线特征

改进点:
1. 使用 PPL (Perplexity) 而不是 Loss
2. 渐进式掩码：每次采样使用相同的掩码集合，从高掩码逐步减少
3. 可选归一化：除以全掩码（100%）情况的 PPL 进行归一化

核心假设:
- Member:  早熟特征 (Early Drop) - 在高掩码率时就快速降低 PPL
- Non-member: 平滑下降 - 需要低掩码率才能降低 PPL

用法:  CUDA_VISIBLE_DEVICES=0 python case_study_loss_trajectory.py
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
from sklearn.metrics import roc_auc_score, roc_curve
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


class PPLTrajectoryAnalyzer:
    """PPL 轨迹分析器（渐进式掩码）"""

    def __init__(self, model, tokenizer, device, max_length=512, num_runs=5, normalize=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_runs = num_runs
        self.normalize = normalize  # 是否除以全掩码 PPL 进行归一化
        self.mask_id, self.shift_logits = get_model_nll_params(model)

        print(f"[PPLTrajectoryAnalyzer] 配置:")
        print(f"  - 渐进式采样次数: {num_runs}")
        print(f"  - 归一化: {normalize}")
        print(f"  - Mask ID: {self.mask_id}, Shift Logits: {self.shift_logits}")

    @torch.no_grad()
    def get_ppl_trajectory(self, text: str, mask_fracs: list) -> dict:
        """
        获取一个文本在不同掩码比例下的 PPL 轨迹（渐进式）

        Args:
            text: 输入文本
            mask_fracs: 掩码比例列表，从高到低 (模拟从 T 到 0)

        Returns:
            {
                'ppls': [ppl_at_t1, ppl_at_t2, ...],  # 对应 mask_fracs
                'normalized_ppls': [... ],  # 归一化后的 PPL
                'mean_ppl': float,
                'features': {...}  # 轨迹特征
            }
        """
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        valid_length = int(attention_mask.sum().item())

        if valid_length == 0:
            return None

        valid_positions = torch.where(attention_mask[0])[0]

        # 多次渐进式采样
        all_trajectories = []  # 存储每次采样的完整轨迹

        for run_idx in range(self.num_runs):
            # === 每次采样都重新随机选择一组掩码位置 ===
            # 基于最高掩码比例确定掩码集合
            max_mask_frac = mask_fracs[0]  # 第一个应该是最高的
            max_num_to_mask = int(round(max_mask_frac * valid_length))
            max_num_to_mask = min(max_num_to_mask, len(valid_positions))

            # 随机选择这组位置（这次采样的所有时间步都用这组）
            perm = torch.randperm(len(valid_positions), device=self.device)
            initial_mask_positions = valid_positions[perm[:max_num_to_mask]]

            # === 渐进式：从高掩码到低掩码，逐步减少掩码数量 ===
            trajectory = []

            for mask_frac in mask_fracs:
                # 当前时间步需要掩码的数量
                num_to_mask = int(round(mask_frac * valid_length))
                num_to_mask = min(num_to_mask, len(initial_mask_positions))

                # 从初始掩码集合中取前 num_to_mask 个
                # （渐进式：保持掩码位置一致，只是数量递减）
                current_mask_positions = initial_mask_positions[:num_to_mask]

                # 创建掩码
                mask = torch.zeros_like(input_ids, dtype=torch.bool)
                if num_to_mask > 0:
                    mask[0, current_mask_positions] = True

                # 掩码输入
                masked_ids = input_ids.clone()
                masked_ids[mask] = self.mask_id

                # 前向传播
                out = self.model(input_ids=masked_ids,
                                 attention_mask=attention_mask if not self.shift_logits else None)
                logits = out.logits if hasattr(out, 'logits') else out[0]

                if self.shift_logits:
                    logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

                # 计算交叉熵损失（只针对掩码位置）
                if num_to_mask > 0:
                    ce = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        input_ids.view(-1),
                        reduction='none'
                    ).view(1, -1).float()

                    avg_loss = ce[mask].mean().item()
                else:
                    # 没有掩码的情况
                    avg_loss = 0.0

                # 转换为 PPL
                # ppl = np.exp(avg_loss)
                trajectory.append(avg_loss)

            all_trajectories.append(trajectory)

        # 对多次采样的轨迹取平均
        all_trajectories = np.array(all_trajectories)  # (num_runs, num_time_steps)
        mean_ppls = np.mean(all_trajectories, axis=0)  # (num_time_steps,)
        std_ppls = np.std(all_trajectories, axis=0)

        # 归一化（可选）
        if self.normalize and mean_ppls[0] > 0:
            # 除以全掩码（第一个时间步）的 PPL
            normalized_ppls = mean_ppls / mean_ppls[0]
        else:
            normalized_ppls = mean_ppls

        # 提取特征（使用归一化或原始 PPL）
        ppls_for_features = normalized_ppls if self.normalize else mean_ppls
        features = self._extract_trajectory_features(ppls_for_features, mask_fracs)

        return {
            'ppls': mean_ppls.tolist(),
            'ppls_std': std_ppls.tolist(),
            'normalized_ppls': normalized_ppls.tolist(),
            'mean_ppl': float(np.mean(mean_ppls)),
            'features': features,
            'all_runs': all_trajectories.tolist()  # 保存所有采样轨迹
        }

    def _extract_trajectory_features(self, ppls: np.ndarray, mask_fracs: list) -> dict:
        """提取 PPL 轨迹的特征"""
        ppls = np.array(ppls)
        mask_fracs = np.array(mask_fracs)

        # 过滤掉 inf 值
        valid_mask = np.isfinite(ppls)
        if not valid_mask.any():
            return self._get_empty_features()

        ppls = ppls[valid_mask]
        mask_fracs = mask_fracs[valid_mask]

        if len(ppls) < 2:
            return self._get_empty_features()

        # 1. 曲线下面积 (AUC of PPL Curve)
        auc = float(np.trapz(ppls, mask_fracs))

        # 2. 总体斜率 (Overall Slope)
        overall_slope = (ppls[-1] - ppls[0]) / (mask_fracs[-1] - mask_fracs[0])

        # 3. 早期斜率 vs 后期斜率
        mid_idx = len(ppls) // 2
        if len(ppls) > 2:
            early_slope = (ppls[mid_idx] - ppls[0]) / (mask_fracs[mid_idx] - mask_fracs[0] + 1e-8)
            late_slope = (ppls[-1] - ppls[mid_idx]) / (mask_fracs[-1] - mask_fracs[mid_idx] + 1e-8)
        else:
            early_slope = overall_slope
            late_slope = overall_slope

        # 4. Early Drop Score - 高掩码时（前1/3）的 PPL 下降程度
        early_third = max(1, len(ppls) // 3)
        early_drop = ppls[0] - ppls[early_third]
        early_drop_normalized = early_drop / (ppls[0] + 1e-8)

        # 5. Early/Late Drop Ratio
        late_drop = ppls[early_third] - ppls[-1] if len(ppls) > early_third else 0
        early_late_ratio = early_drop / (late_drop + 1e-8)

        # 6. 曲率 (Curvature)
        if len(ppls) > 2:
            first_derivative = np.diff(ppls)
            second_derivative = np.diff(first_derivative)
            curvature = float(np.mean(np.abs(second_derivative)))
        else:
            curvature = 0

        # 7. PPL 变异系数 (Coefficient of Variation)
        cv = float(np.std(ppls) / (np.mean(ppls) + 1e-8))

        # 8. 最大下降点
        if len(ppls) > 1:
            drops = -np.diff(ppls)  # PPL 应该下降
            max_drop_idx = int(np.argmax(drops))
            max_drop_value = float(drops[max_drop_idx])
            max_drop_position = float(mask_fracs[max_drop_idx])

            # 加权下降位置
            drop_probs = drops / (drops.sum() + 1e-8)
            weighted_drop_position = float(np.sum(drop_probs * mask_fracs[:-1]))
        else:
            max_drop_idx = 0
            max_drop_value = 0
            max_drop_position = mask_fracs[0]
            weighted_drop_position = mask_fracs[0]

        # 9. 高掩码区域平均 PPL（前1/3）
        high_mask_avg_ppl = float(np.mean(ppls[: early_third]))

        # 10. 低掩码区域平均 PPL（后1/3）
        late_third = max(1, len(ppls) * 2 // 3)
        low_mask_avg_ppl = float(np.mean(ppls[late_third:]))

        return {
            'auc': auc,
            'overall_slope': float(overall_slope),
            'early_slope': float(early_slope),
            'late_slope': float(late_slope),
            'early_drop': float(early_drop),
            'early_drop_normalized': float(early_drop_normalized),
            'early_late_ratio': float(early_late_ratio),
            'curvature': float(curvature),
            'cv': float(cv),
            'max_drop_idx': max_drop_idx,
            'max_drop_value': float(max_drop_value),
            'max_drop_position': float(max_drop_position),
            'weighted_drop_position': float(weighted_drop_position),
            'high_mask_avg_ppl': float(high_mask_avg_ppl),
            'low_mask_avg_ppl': float(low_mask_avg_ppl),
        }

    def _get_empty_features(self):
        """返回空特征（用于异常情况）"""
        return {
            'auc': 0,
            'overall_slope': 0,
            'early_slope': 0,
            'late_slope': 0,
            'early_drop': 0,
            'early_drop_normalized': 0,
            'early_late_ratio': 0,
            'curvature': 0,
            'cv': 0,
            'max_drop_idx': 0,
            'max_drop_value': 0,
            'max_drop_position': 0,
            'weighted_drop_position': 0,
            'high_mask_avg_ppl': 0,
            'low_mask_avg_ppl': 0,
        }


def analyze_trajectories(analyzer, dataset, mask_fracs, num_samples=100):
    """分析数据集的 PPL 轨迹"""
    # 采样
    member_idx = [i for i, x in enumerate(dataset) if x['label'] == 1]
    non_member_idx = [i for i, x in enumerate(dataset) if x['label'] == 0]

    np.random.seed(42)
    sel_member = np.random.choice(member_idx, min(num_samples, len(member_idx)), replace=False)
    sel_non = np.random.choice(non_member_idx, min(num_samples, len(non_member_idx)), replace=False)

    # 分析 Member
    member_results = []
    print("\n分析 Member 样本的 PPL 轨迹...")
    for idx in tqdm(sel_member, desc="Member"):
        result = analyzer.get_ppl_trajectory(dataset[int(idx)]['text'], mask_fracs)
        if result:
            member_results.append(result)

    # 分析 Non-member
    non_member_results = []
    print("分析 Non-member 样本的 PPL 轨迹...")
    for idx in tqdm(sel_non, desc="Non-Member"):
        result = analyzer.get_ppl_trajectory(dataset[int(idx)]['text'], mask_fracs)
        if result:
            non_member_results.append(result)

    return {
        'member': member_results,
        'non_member': non_member_results,
        'mask_fracs': mask_fracs
    }


def plot_trajectory_analysis(results, output_dir, normalize):
    """绘制 PPL 轨迹分析结果"""
    os.makedirs(output_dir, exist_ok=True)

    member_results = results['member']
    non_member_results = results['non_member']
    mask_fracs = results['mask_fracs']

    # 选择使用归一化或原始 PPL
    ppl_key = 'normalized_ppls' if normalize else 'ppls'
    y_label = 'Normalized PPL' if normalize else 'PPL'

    # 提取所有轨迹
    member_trajectories = np.array([r[ppl_key] for r in member_results])
    non_member_trajectories = np.array([r[ppl_key] for r in non_member_results])

    # 过滤掉异常值
    member_trajectories = np.clip(member_trajectories, 0, 1000)
    non_member_trajectories = np.clip(non_member_trajectories, 0, 1000)

    # 创建大图
    fig = plt.figure(figsize=(20, 14))

    # === 子图 1: PPL 轨迹曲线 ===
    ax1 = plt.subplot(3, 4, 1)

    member_mean = member_trajectories.mean(axis=0)
    member_std = member_trajectories.std(axis=0)
    non_member_mean = non_member_trajectories.mean(axis=0)
    non_member_std = non_member_trajectories.std(axis=0)

    x = np.array(mask_fracs) * 100

    ax1.plot(x, member_mean, 'o-', linewidth=2.5, markersize=8,
             color='#2E86AB', label='Member (Mean)')
    ax1.fill_between(x, member_mean - member_std, member_mean + member_std,
                     alpha=0.3, color='#2E86AB')

    ax1.plot(x, non_member_mean, 's-', linewidth=2.5, markersize=8,
             color='#A23B72', label='Non-Member (Mean)')
    ax1.fill_between(x, non_member_mean - non_member_std, non_member_mean + non_member_std,
                     alpha=0.3, color='#A23B72')

    ax1.set_xlabel('Mask Ratio (%)', fontweight='bold', fontsize=11)
    ax1.set_ylabel(y_label, fontweight='bold', fontsize=11)
    ax1.set_title(f'{y_label} Trajectory (T→0)', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # === 子图 2: 个体轨迹样本 ===
    ax2 = plt.subplot(3, 4, 2)

    n_show = min(15, len(member_trajectories))
    for i in np.random.choice(len(member_trajectories), n_show, replace=False):
        ax2.plot(x, member_trajectories[i], alpha=0.3, color='#2E86AB', linewidth=1)

    for i in np.random.choice(len(non_member_trajectories), n_show, replace=False):
        ax2.plot(x, non_member_trajectories[i], alpha=0.3, color='#A23B72', linewidth=1)

    ax2.plot(x, member_mean, linewidth=3, color='#2E86AB', label='Member Mean', zorder=10)
    ax2.plot(x, non_member_mean, linewidth=3, color='#A23B72', label='Non-Member Mean', zorder=10)

    ax2.set_xlabel('Mask Ratio (%)', fontweight='bold', fontsize=11)
    ax2.set_ylabel(y_label, fontweight='bold', fontsize=11)
    ax2.set_title('Individual Trajectories', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # === 子图 3-12: 特征分布对比 ===
    feature_names = [
        ('early_drop_normalized', 'Early Drop Score'),
        ('early_late_ratio', 'Early/Late Drop Ratio'),
        ('overall_slope', 'Overall Slope'),
        ('curvature', 'Curvature'),
        ('weighted_drop_position', 'Weighted Drop Position (%)'),
        ('cv', 'Coefficient of Variation'),
        ('high_mask_avg_ppl', 'High Mask Avg PPL'),
        ('low_mask_avg_ppl', 'Low Mask Avg PPL'),
        ('auc', 'AUC of PPL Curve'),
        ('early_slope', 'Early Slope'),
    ]

    for idx, (feat_key, feat_name) in enumerate(feature_names, start=3):
        ax = plt.subplot(3, 4, idx)

        member_feat = [r['features'][feat_key] for r in member_results]
        non_member_feat = [r['features'][feat_key] for r in non_member_results]

        # 转换百分比
        if 'position' in feat_key:
            member_feat = [x * 100 for x in member_feat]
            non_member_feat = [x * 100 for x in non_member_feat]

        # 过滤异常值
        member_feat = [x for x in member_feat if np.isfinite(x) and abs(x) < 1e6]
        non_member_feat = [x for x in non_member_feat if np.isfinite(x) and abs(x) < 1e6]

        if not member_feat or not non_member_feat:
            continue

        # 绘制直方图
        ax.hist(member_feat, bins=20, alpha=0.6, label='Member',
                color='#2E86AB', edgecolor='black')
        ax.hist(non_member_feat, bins=20, alpha=0.6, label='Non-Member',
                color='#A23B72', edgecolor='black')

        ax.axvline(np.mean(member_feat), color='#2E86AB', linestyle='--', linewidth=2)
        ax.axvline(np.mean(non_member_feat), color='#A23B72', linestyle='--', linewidth=2)

        # 计算 AUROC
        all_feat = np.concatenate([member_feat, non_member_feat])
        all_labels = np.concatenate([np.ones(len(member_feat)), np.zeros(len(non_member_feat))])

        # 根据特征决定方向
        if feat_key in ['auc', 'cv', 'high_mask_avg_ppl']:
            try:
                auroc = roc_auc_score(all_labels, -np.array(all_feat))
            except:
                auroc = 0.5
        else:
            try:
                auroc = roc_auc_score(all_labels, np.array(all_feat))
            except:
                auroc = 0.5

        ax.set_xlabel(feat_name, fontweight='bold', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{feat_name}\n(AUROC:  {auroc:.4f})', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ppl_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ PPL 轨迹分析图已保存:  {output_dir}/ppl_trajectory_analysis.png")
    plt.close()


def plot_roc_curves(results, output_dir):
    """绘制各特征的 ROC 曲线"""
    os.makedirs(output_dir, exist_ok=True)

    member_results = results['member']
    non_member_results = results['non_member']

    feature_keys = ['early_drop_normalized', 'early_late_ratio', 'overall_slope',
                    'curvature', 'weighted_drop_position', 'cv',
                    'high_mask_avg_ppl', 'low_mask_avg_ppl', 'auc', 'early_slope']
    feature_names = ['Early Drop', 'Early/Late Ratio', 'Slope', 'Curvature',
                     'Weighted Drop Pos', 'CV', 'High Mask PPL', 'Low Mask PPL', 'AUC', 'Early Slope']

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    best_auroc = 0
    best_feature = None
    auroc_scores = {}

    for idx, (feat_key, feat_name) in enumerate(zip(feature_keys, feature_names)):
        ax = axes[idx]

        member_feat = [r['features'][feat_key] for r in member_results]
        non_member_feat = [r['features'][feat_key] for r in non_member_results]

        # 过滤异常值
        member_feat = [x for x in member_feat if np.isfinite(x) and abs(x) < 1e6]
        non_member_feat = [x for x in non_member_feat if np.isfinite(x) and abs(x) < 1e6]

        if not member_feat or not non_member_feat:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            ax.set_title(feat_name)
            continue

        all_feat = np.concatenate([member_feat, non_member_feat])
        all_labels = np.concatenate([np.ones(len(member_feat)), np.zeros(len(non_member_feat))])

        # 根据特征决定方向
        if feat_key in ['auc', 'cv', 'high_mask_avg_ppl']:
            scores = -np.array(all_feat)
        else:
            scores = np.array(all_feat)

        try:
            fpr, tpr, _ = roc_curve(all_labels, scores)
            auroc = roc_auc_score(all_labels, scores)
            auroc_scores[feat_name] = auroc

            if auroc > best_auroc:
                best_auroc = auroc
                best_feature = feat_name

            ax.plot(fpr, tpr, linewidth=2.5, label=f'AUC={auroc:.4f}', color='#2E86AB')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlabel('FPR', fontweight='bold')
            ax.set_ylabel('TPR', fontweight='bold')
            ax.set_title(f'{feat_name}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:  {str(e)[: 20]}', ha='center', va='center')
            ax.set_title(feat_name)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ ROC 曲线已保存: {output_dir}/roc_curves.png")
    if best_feature:
        print(f"🏆 最佳特征:  {best_feature} (AUROC: {best_auroc:.4f})")
    plt.close()

    return best_feature, best_auroc, auroc_scores


def save_results(results, output_dir, normalize):
    """保存结果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)

    member_results = results['member']
    non_member_results = results['non_member']

    feature_keys = ['early_drop_normalized', 'early_late_ratio', 'overall_slope',
                    'curvature', 'weighted_drop_position', 'cv',
                    'high_mask_avg_ppl', 'low_mask_avg_ppl', 'auc', 'early_slope']

    summary = {
        'config': {
            'mask_fracs': results['mask_fracs'],
            'normalize': normalize
        },
        'num_samples': {
            'member': len(member_results),
            'non_member': len(non_member_results)
        },
        'features': {}
    }

    for feat_key in feature_keys:
        member_feat = [r['features'][feat_key] for r in member_results
                       if np.isfinite(r['features'][feat_key]) and abs(r['features'][feat_key]) < 1e6]
        non_member_feat = [r['features'][feat_key] for r in non_member_results
                           if np.isfinite(r['features'][feat_key]) and abs(r['features'][feat_key]) < 1e6]

        if not member_feat or not non_member_feat:
            continue

        all_feat = np.concatenate([member_feat, non_member_feat])
        all_labels = np.concatenate([np.ones(len(member_feat)), np.zeros(len(non_member_feat))])

        if feat_key in ['auc', 'cv', 'high_mask_avg_ppl']:
            try:
                auroc = roc_auc_score(all_labels, -np.array(all_feat))
            except:
                auroc = 0.5
        else:
            try:
                auroc = roc_auc_score(all_labels, np.array(all_feat))
            except:
                auroc = 0.5

        try:
            t_stat, p_value = stats.ttest_ind(member_feat, non_member_feat)
        except:
            p_value = 1.0

        summary['features'][feat_key] = {
            'member_mean': float(np.mean(member_feat)),
            'member_std': float(np.std(member_feat)),
            'non_member_mean': float(np.mean(non_member_feat)),
            'non_member_std': float(np.std(non_member_feat)),
            'auroc': float(auroc),
            'p_value': float(p_value)
        }

    with open(f'{output_dir}/ppl_trajectory_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ 统计结果已保存: {output_dir}/ppl_trajectory_summary.json")


def main():
    parser = argparse.ArgumentParser(description="PPL Trajectory Shape Analysis")
    parser.add_argument('-c', '--config', default='attack/configs/config_fond.yaml')
    parser.add_argument('--mask-fracs', type=float, nargs='+',
                        default=[1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0],
                        help='掩码比例序列 (从高到低，模拟 T→0)')
    parser.add_argument('--num-samples', type=int, default=2000)
    parser.add_argument('--num-runs', type=int, default=1,
                        help='每个样本的渐进式采样次数')
    parser.add_argument('--normalize', action='store_true',
                        help='是否除以全掩码(100%%)的PPL进行归一化')
    parser.add_argument('--output', default='./ppl_trajectory_output')
    args = parser.parse_args()

    print("=" * 80)
    print("PPL Trajectory Shape Analysis (渐进式掩码)")
    print("=" * 80)
    print(f"\n改进点:")
    print("  ✓ 使用 PPL 而不是 Loss")
    print("  ✓ 渐进式掩码 - 每次采样使用相同的掩码集合")
    print(f"  ✓ 归一化:  {args.normalize}")
    print(f"\n假设验证:")
    print("  ✓ Member:  早熟特征 (Early Drop) - 高掩码时就快速降低 PPL")
    print("  ✓ Non-member: 平滑下降 - 需要低掩码才能降低 PPL")

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    global_config = config['global']

    set_seed(global_config.get('seed', 42))

    # 加载模型
    model_path = global_config['target_model']
    print(f"\n📥 加载模型: {model_path}")
    ModelManager.register_custom_models()

    device = torch.device(global_config.get('device', 'cuda'))
    model, tokenizer, device = ModelManager.init_model(model_path, model_path, device)
    print("✅ 模型加载完成")

    # 加载数据
    ds_config = global_config['datasets'][0]
    train_path = ds_config['json_train_path']
    test_path = ds_config['json_test_path']
    dataset = load_local_json_dataset(train_path, test_path)

    # 确保 mask_fracs 从高到低排序
    mask_fracs = sorted(args.mask_fracs, reverse=True)
    print(f"\n🔬 PPL 轨迹分析 (T→0)")
    print(f"掩码比例序列: {[f'{x:.0%}' for x in mask_fracs]}")
    print(f"每类样本数: {args.num_samples}, 渐进式采样次数:  {args.num_runs}\n")

    # 分析
    analyzer = PPLTrajectoryAnalyzer(model, tokenizer, device,
                                     global_config.get('max_length', 512),
                                     args.num_runs, args.normalize)

    results = analyze_trajectories(analyzer, dataset, mask_fracs, args.num_samples)

    # 可视化
    print("\n🎨 生成可视化...")
    plot_trajectory_analysis(results, args.output, args.normalize)
    best_feature, best_auroc, auroc_scores = plot_roc_curves(results, args.output)

    # 保存结果
    save_results(results, args.output, args.normalize)

    # 打印摘要
    print("\n" + "=" * 80)
    print("📊 分析摘要")
    print("=" * 80)

    member_results = results['member']
    non_member_results = results['non_member']

    # 打印前 3 个最佳特征
    if auroc_scores:
        sorted_features = sorted(auroc_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Top 3 特征:")
        for i, (feat_name, auroc) in enumerate(sorted_features[:3], 1):
            print(f"  {i}.{feat_name}: AUROC = {auroc:.4f}")

    # Early Drop Score
    member_early = [r['features']['early_drop_normalized'] for r in member_results
                    if np.isfinite(r['features']['early_drop_normalized'])]
    non_early = [r['features']['early_drop_normalized'] for r in non_member_results
                 if np.isfinite(r['features']['early_drop_normalized'])]

    if member_early and non_early:
        print(f"\n✨ Early Drop Score:")
        print(f"  Member:       {np.mean(member_early):.6f} ± {np.std(member_early):.6f}")
        print(f"  Non-Member: {np.mean(non_early):.6f} ± {np.std(non_early):.6f}")
        print(f"  差异:  {np.mean(member_early) - np.mean(non_early):.6f}")

    print("\n" + "=" * 80)
    print(f"✅ 分析完成!  结果保存在: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()