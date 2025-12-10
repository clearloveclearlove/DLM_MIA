import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params


class TestAttack(AbstractAttack):
    """
    DLM Convergence-Variance Attack

    思路：
      - 对一个样本的后半句（或后80%）做掩码，记初始掩码集合 S（大小为 M）。
      - 进行 K 步推理（逐步去掩码）：
          每一步：用当前序列前向 -> 取初始掩码集合 S 中每个位置的 token 对应的 CE loss，
                 记录为第 t 步的列，得到 (M × K) 矩阵；
                 同时按置信度（softmax max prob）替换其中一部分 mask 为模型预测，进入下一步。
      - 计算 (M × K) 矩阵按列（时间维）对每个 token 的方差，再对所有 token 取平均，得到 mean_variance。
      - 评分：score = 1 / (1 + mean_variance)。成员 → 收敛快 → 方差低 → 分数高。
    
    可配置参数（config）：
      name: 新增列名（默认 "dlm_conv_var"）
      steps: K，步数（默认 10）
      batch_size: 批大小（默认 8）
      max_length: 最大长度（默认 512）
      mask_tail_fraction: 掩码尾部比例（默认 0.5，表示掩码后半句；设置 0.8 表示掩码后 80%）
      per_step_unmask_frac: 每步替换的掩码比例（默认 None → 自动按 K 均分；例如 0.2 表示每步替换 20%）
      choose_strategy: "confidence" 或 "uniform"（默认 "confidence"：优先替换置信度高的 mask）
      save_metadata: 是否保存元数据（默认 True）
      metadata_dir: 元数据根目录（默认 "./"）
      seed: 随机种子（默认 42）
    """

    def __init__(self, name: str, model, tokenizer, config: Dict[str, Any], device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # —— 基本参数 ——
        self.steps = int(config.get("steps", 10))
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 512))

        # —— 掩码范围：只掩码尾部 —— 
        self.mask_tail_fraction = float(config.get("mask_tail_fraction", 0.5))  # 0.5=后半句；0.8=后80%
        self.mask_tail_fraction = min(max(self.mask_tail_fraction, 0.0), 1.0)

        # —— 每步替换比例（未设定则按 K 均匀替换，确保 K 步填满）——
        self.per_step_unmask_frac = config.get("per_step_unmask_frac", None)
        if self.per_step_unmask_frac is not None:
            self.per_step_unmask_frac = float(self.per_step_unmask_frac)
            self.per_step_unmask_frac = min(max(self.per_step_unmask_frac, 1e-6), 1.0)

        # —— 选择/替换策略 ——
        self.choose_strategy = str(config.get("choose_strategy", "confidence"))  # "confidence" or "uniform"

        # —— 元数据保存 ——
        self.save_metadata = bool(config.get("save_metadata", True))
        self.metadata_dir = os.path.join(
            str(config.get("metadata_dir", "./")),
            f"dlm_conv_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if self.save_metadata:
            os.makedirs(self.metadata_dir, exist_ok=True)
            with open(os.path.join(self.metadata_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)
        self.metadata_buffer = []

        # —— 取模型的 mask_id / shift_logits 行为（与你的工程工具保持一致）——
        if "model_mask_id" in config and "model_shift_logits" in config:
            self.target_mask_id = int(config["model_mask_id"])
            self.target_shift_logits = bool(config["model_shift_logits"])
        else:
            self.target_mask_id, self.target_shift_logits = get_model_nll_params(self.model)

        # —— 随机性 —— 
        self.seed = int(config.get("seed", 42))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # —— 输出列名（数据集上新增的分数列）——
        self.output_col = str(config.get("name", self.name or "dlm_conv_var"))

    # ------------------------------ Public API ------------------------------

    def run(self, dataset: Dataset) -> Dataset:
        n = len(dataset)
        if n == 0:
            return dataset

        scores = []
        for start in tqdm(range(0, n, self.batch_size), desc=f"{self.output_col}"):
            end = min(start + self.batch_size, n)
            batch = dataset[start:end]
            texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
            batch_scores = self._score_batch(texts, batch_start_idx=start)
            scores.extend(batch_scores)

        dataset = dataset.add_column(self.output_col, scores)

        if self.save_metadata and self.metadata_buffer:
            path = os.path.join(self.metadata_dir, "full_metadata.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.metadata_buffer, f, indent=2, ensure_ascii=False)
            print(f"[DLMConvergenceAttack] Metadata saved to {path}")

        return dataset

    # --------------------------- Internals -----------------------------

    @torch.no_grad()
    def _score_batch(self, texts: List[str], batch_start_idx: int) -> List[float]:
        B = len(texts)

        # Tokenize
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        input_ids = encoded["input_ids"]                       # (B, L)
        attention_mask = encoded["attention_mask"].bool()      # (B, L)
        L = input_ids.size(1)

        # 初始掩码集合：只掩码尾部 mask_tail_fraction 的有效 token
        init_mask = self._build_mask(attention_mask, fraction=self.mask_tail_fraction)  # (B, L) bool
        # 记录初始掩码集合 S（用于构造 M×K 矩阵）
        init_mask_positions = [torch.where(init_mask[b])[0] for b in range(B)]  # list of 1D indices

        # 序列 x 随步骤逐步替换
        x = input_ids.clone()
        x[init_mask] = self.target_mask_id

        # 用于记录 loss 矩阵：每个样本单独长度，用 list 装 (M_b × K) numpy
        loss_mats: List[np.ndarray] = [np.full((len(init_mask_positions[b]), self.steps), np.nan, dtype=np.float32)
                                       for b in range(B)]

        # step 解码记录
        step_texts: List[List[str]] = [[] for _ in range(B)]

        # 逐步去掩码
        remaining_mask = init_mask.clone()  # (B, L)
        for step in range(self.steps):
            out = self.model(
                input_ids=x,
                attention_mask=attention_mask if not self.target_shift_logits else None
            )
            logits = out.logits if hasattr(out, "logits") else out[0]  # (B, L, V)

            # 需要 shift logits 的模型：对齐标签
            if self.target_shift_logits:
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

            # 计算：对“初始掩码集合 S 的所有位置” 的 CE（即便某些位置这一步已经被替换，也照样评估 loss）
            ce_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction="none"
            ).view(B, L).float()

            # 写入每个样本的第 step 列
            for b in range(B):
                pos = init_mask_positions[b]
                if pos.numel() > 0:
                    loss_mats[b][:, step] = ce_all[b, pos].detach().cpu().numpy()

            # 记录当前解码文本
            pred_ids = logits.argmax(dim=-1)  # (B, L)
            x_show = x.clone()
            # 仅为了可读展示，把当前仍是 mask 的位置临时显示为预测结果
            x_show[remaining_mask] = pred_ids[remaining_mask]
            for b in range(B):
                step_texts[b].append(self.tokenizer.decode(x_show[b], skip_special_tokens=False))

            # 选择要替换的一部分 mask 位置（真正“去掩码”）
            to_unmask = self._choose_positions_to_unmask(
                logits=logits,
                remaining_mask=remaining_mask,
                per_step_unmask_frac=self.per_step_unmask_frac,
                steps_total=self.steps,
                step_idx=step,
                strategy=self.choose_strategy
            )
            # 用预测 token 覆盖这些位置
            if to_unmask is not None:
                rows, cols = to_unmask
                x[rows, cols] = pred_ids[rows, cols]
                remaining_mask[rows, cols] = False

            # 如果所有 mask 都替完，提前结束
            if not remaining_mask.any():
                # 补齐后续步的 loss 列（如果有），就复制最后一列，保持矩阵维度整齐
                for b in range(B):
                    if loss_mats[b].shape[1] > (step + 1):
                        last_col = loss_mats[b][:, step][:, None]
                        loss_mats[b][:, step + 1:] = last_col
                break

        # 计算每个样本的 mean variance（沿步骤维度对每个 token 取方差，再取平均）
        scores = []
        batch_meta = []
        for b in range(B):
            mat = loss_mats[b]  # (M_b, K)
            if mat.size == 0:
                mean_var = float("inf")  # 没有掩码就没法评估
                score = 0.0
            else:
                # 方差沿列（时间）方向
                var_each = np.var(mat, axis=1)  # (M_b,)
                mean_var = float(np.mean(var_each))
                # 得分：方差小→收敛快→score高；一个简单单调变换
                # score = float(1.0 / (1.0 + mean_var))
                score = mean_var

            scores.append(score)

            # 元数据
            if self.save_metadata:
                valid_len = int(attention_mask[b].sum().item())
                meta = {
                    "sample_idx": batch_start_idx + b,
                    "text": texts[b][:200],
                    "valid_length": valid_len,
                    "num_masked_init": int(len(init_mask_positions[b])),
                    "steps": self.steps,
                    "mean_variance": mean_var,
                    "score": score,
                    "decode_per_step": step_texts[b],  # 每步可读文本
                    # 只保存 loss 矩阵摘要，避免文件过大
                    "loss_matrix_summary": {
                        "shape": [int(s) for s in mat.shape],
                        "first_row": mat[0].tolist() if mat.shape[0] > 0 else [],
                        "col_mean": np.mean(mat, axis=0).tolist() if mat.shape[0] > 0 else [],
                        "col_std": np.std(mat, axis=0).tolist() if mat.shape[0] > 0 else [],
                    },
                }
                batch_meta.append(meta)

        if self.save_metadata and batch_meta:
            self.metadata_buffer.extend(batch_meta)

        return scores

    # --------------------------- Helpers -----------------------------

    @staticmethod
    def _build_mask(
        attention_mask: torch.Tensor,
        *,
        strategy: str = "random",          # "tail" | "random"
        fraction: float = 0.5,           # 掩码比例（0~1），strategy="tail"/"random" 都可用
        count: int = None,               # 固定掩码数量（优先级高于 fraction），仅当 strategy="random" 时常用
        seed: int = 42,                # 可复现随机
    ) -> torch.Tensor:
        """
        依据策略生成掩码矩阵（True=mask）。
        attention_mask: (B, L) bool
        strategy:
        - "tail":   只掩码每个样本有效 token 的尾部 'fraction' 比例
        - "random": 在每个样本有效 token 中随机挑选 'count' 个或 'fraction' 比例进行掩码
        """
        assert strategy in ("tail", "random"), f"unknown mask strategy: {strategy}"
        B, L = attention_mask.shape
        result = torch.zeros_like(attention_mask, dtype=torch.bool)

        # 随机生成器（保持可复现）
        g = None
        if seed is not None:
            g = torch.Generator(device=attention_mask.device)
            g.manual_seed(seed)

        for b in range(B):
            valid_pos = torch.where(attention_mask[b])[0]  # 有效 token 的位置
            n = int(valid_pos.numel())
            if n == 0:
                continue

            if strategy == "tail":
                k = int(round(n * fraction))
                if k <= 0:
                    continue
                tail_idx = valid_pos[-k:]  # 尾部 k 个
                result[b, tail_idx] = True

            elif strategy == "random":
                if count is not None:
                    k = min(max(int(count), 0), n)
                else:
                    k = int(round(n * fraction))
                if k <= 0:
                    continue
                # 在有效 token 中随机选择 k 个
                perm = torch.randperm(n, generator=g, device=attention_mask.device)
                chosen = valid_pos[perm[:k]]
                result[b, chosen] = True

        return result

    def _choose_positions_to_unmask(
        self,
        logits: torch.Tensor,                # (B, L, V)
        remaining_mask: torch.Tensor,        # (B, L) bool
        per_step_unmask_frac: float,
        steps_total: int,
        step_idx: int,
        strategy: str = "confidence",
    ):
        """
        返回 (rows, cols) 需要在本步“真正替换的 mask 位置”。
        - strategy="confidence": 优先替换 softmax 最大概率高的 mask
        - strategy="uniform": 在剩余 mask 中均匀随机
        - per_step_unmask_frac 未提供时，自动按 K 均匀替换（确保 K 步替完）
        """
        B, L, V = logits.shape

        # 剩余 mask 位置列表
        rows_list, cols_list = torch.where(remaining_mask)
        if rows_list.numel() == 0:
            return None

        # 计算本步替换数量
        # 如果没有给 per_step_unmask_frac，就尽量 K 步均匀替完
        if per_step_unmask_frac is None:
            # 估算剩余数量
            remain_cnt = int(remaining_mask.sum().item())
            # 剩余步数（包含当前步）
            remain_steps = max(1, steps_total - step_idx)
            num_to_unmask = max(1, remain_cnt // remain_steps)
        else:
            total_remain = int(remaining_mask.sum().item())
            num_to_unmask = max(1, int(round(total_remain * per_step_unmask_frac)))

        num_to_unmask = min(num_to_unmask, rows_list.numel())

        if strategy == "uniform":
            perm = torch.randperm(rows_list.numel(), device=logits.device)
            choose = perm[:num_to_unmask]
            return rows_list[choose], cols_list[choose]
        else:
            # confidence：softmax 最大概率高者优先
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)  # (B, L, V)
                max_probs = probs[rows_list, cols_list].max(dim=-1).values  # (N_mask,)
                top_idx = torch.topk(max_probs, k=num_to_unmask, largest=True).indices
            return rows_list[top_idx], cols_list[top_idx]
