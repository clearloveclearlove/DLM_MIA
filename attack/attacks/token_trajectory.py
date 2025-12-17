

import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params
from attack.misc.models import ModelManager


class TokenTrajectoryAttack(AbstractAttack):
    """
    Token-Level Trajectory AUC Attack (Corrected & Vectorized).

    Logic:
    1. Computes probability trajectories of ground-truth tokens as context is progressively unmasked.
    2. Calculates AUC for each token's trajectory.
    3. Aggregates scores using 'bottom_k' (focusing on the hardest tokens).

    Hypothesis (Fixed):
    - Members: Even on 'hard' tokens, the model has higher confidence due to memorization -> Higher AUC.
    - Non-members: Hard tokens remain hard (low probability) until full context is given -> Lower AUC.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # Core parameters
        self.num_steps = int(config.get("num_steps", 6))  # Context unmasking steps
        self.num_bins = int(config.get("num_bins", 3))  # Number of token bins
        self.batch_size = int(config.get("batch_size", 4))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))

        # Aggregation method for token-level scores
        self.aggregation = config.get("aggregation", "bottom_k")
        self.aggregation_ratio = float(config.get("aggregation_ratio", 0.01))  # For top_k/bottom_k

        # Use reference model for calibration
        self.use_reference_model = config.get("use_reference_model", False)
        self.ref_model = None
        self.ref_tokenizer = None

        # Model-specific parameters
        if 'model_mask_id' in config and 'model_shift_logits' in config:
            self.target_mask_id = config['model_mask_id']
            self.target_shift_logits = config['model_shift_logits']
        else:
            self.target_mask_id, self.target_shift_logits = get_model_nll_params(self.model)

        # Initialize reference model if needed
        if self.use_reference_model:
            self._init_reference_model(config)

        # Generate context mask schedule
        self.context_mask_ratios = self._generate_mask_schedule()

        logging.info(f"[TokenTrajectoryAttack - Fully Vectorized] Initialized:")
        logging.info(f"  - num_steps: {self.num_steps}")
        logging.info(f"  - num_bins:  {self.num_bins}")
        logging.info(f"  - batch_size: {self.batch_size}")
        logging.info(f"  - aggregation: {self.aggregation} (ratio={self.aggregation_ratio})")
        logging.info(f"  - use_reference_model: {self.use_reference_model}")
        logging.info(f"  - context_mask_ratios: {[f'{r:.2f}' for r in self.context_mask_ratios]}")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _init_reference_model(self, config):
        """Initialize reference model for calibration."""
        ref_model_path = config.get('reference_model_path')
        if not ref_model_path:
            raise ValueError("reference_model_path must be specified when use_reference_model=True")

        logging.info(f"Loading reference model from: {ref_model_path}")

        ref_device = torch.device(config.get("reference_device", "cuda"))
        self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
            ref_model_path, ref_model_path, ref_device
        )
        self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.ref_model)

        # Get actual device after DataParallel wrapping
        if isinstance(self.ref_model, torch.nn.DataParallel):
            self.ref_device = next(self.ref_model.parameters()).device
        else:
            self.ref_device = ref_device

        logging.info(f"Reference model loaded on {self.ref_device}")

    def _generate_mask_schedule(self):
        """Generate progressive context mask ratios from 1. 0 to 0.0."""
        return np.linspace(1.0, 0.0, self.num_steps)

    def run(self, dataset: Dataset) -> Dataset:
        """Run token-level trajectory attack on the dataset."""
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset

        membership_scores = []

        # Process in batches
        for start_idx in tqdm(range(0, n_samples, self.batch_size), desc=f"{self.name}"):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = dataset[start_idx:end_idx]

            # 🔥 Fully vectorized batch processing
            batch_scores = self._compute_batch_scores(batch["text"])
            membership_scores.extend(batch_scores)

        # Cleanup reference model
        if self.ref_model is not None:
            self.ref_model.to('cpu')
            del self.ref_model
            del self.ref_tokenizer
            torch.cuda.empty_cache()

        return dataset.add_column(self.name, membership_scores)

    def _compute_batch_scores(self, texts):
        """
        Compute scores.
        Higher Score = Higher Likelihood of Membership.
        """
        if self.use_reference_model:
            target_aucs = self._compute_token_aucs_batch(
                texts, self.model, self.tokenizer, self.target_mask_id,
                self.target_shift_logits, self.device
            )
            ref_aucs = self._compute_token_aucs_batch(
                texts, self.ref_model, self.ref_tokenizer, self.ref_mask_id,
                self.ref_shift_logits, self.ref_device
            )

            scores = []
            for t_auc, r_auc in zip(target_aucs, ref_aucs):
                if t_auc is None or r_auc is None:
                    scores.append(0.0)
                    continue

                # Aggregate using bottom_k (find the hardest tokens)
                t_agg = self._aggregate_scores(t_auc)
                r_agg = self._aggregate_scores(r_auc)

                # Ratio: Target AUC / Ref AUC
                # Member: Target AUC is high (even for hard tokens), Ref is baseline. Ratio > 1.
                scores.append(t_agg / max(r_agg, 1e-6))  # ❌ Removed negative sign
        else:
            target_aucs = self._compute_token_aucs_batch(
                texts, self.model, self.tokenizer, self.target_mask_id,
                self.target_shift_logits, self.device
            )
            scores = []
            for t_auc in target_aucs:
                if t_auc is None:
                    scores.append(0.0)
                    continue
                # Higher AUC on hard tokens -> Member
                scores.append(self._aggregate_scores(t_auc))  # ❌ Removed negative sign

        return scores

    def _compute_token_aucs_batch(self, texts, model, tokenizer, mask_id, shift_logits, device):
        """
        Compute token-level trajectory AUCs for a batch of texts using Micro-Batching.

        🚀 核心修复:
        不一次性堆叠所有 Step (避免 Gather OOM)，而是切分成显卡能吃得消的小块 (Micro-Batch)。
        """
        # =========================================================================
        # 🔥 关键参数：微批次大小
        # 4张卡时，设为 8 表示每张卡处理 2 个样本。
        # 如果依然 OOM，请将此值改为 4。
        MICRO_BATCH_SIZE = 8
        # =========================================================================

        batch_size = len(texts)

        # 1. 批量 Tokenize
        try:
            encoded = tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            ).to(device)
        except Exception as e:
            logging.warning(f"Tokenization error: {e}")
            return [None] * batch_size

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        valid_lengths = attention_mask.sum(dim=1)

        # 2. 准备 Bin 元数据
        batch_bin_metadata = []
        for b in range(batch_size):
            valid_len = int(valid_lengths[b].item())
            if valid_len < 2:
                batch_bin_metadata.append(None)
                continue

            all_valid_positions = list(range(1, valid_len - 1))
            rng = random.Random(self.seed + b)
            rng.shuffle(all_valid_positions)

            if not all_valid_positions:
                batch_bin_metadata.append(None)
                continue

            num_bins = min(self.num_bins, len(all_valid_positions))
            bin_size = len(all_valid_positions) // num_bins

            bins = []
            for bin_idx in range(num_bins):
                start_idx = bin_idx * bin_size
                end_idx = len(all_valid_positions) if bin_idx == num_bins - 1 else (bin_idx + 1) * bin_size
                bin_positions = all_valid_positions[start_idx:end_idx]
                if not bin_positions: continue

                context_positions = [p for p in range(valid_len) if p not in bin_positions]

                # 在 CPU 上打乱 Context，节省 GPU 显存
                g = torch.Generator()
                g.manual_seed(self.seed + bin_idx + b * 100)
                context_perm = torch.randperm(len(context_positions), generator=g)
                context_positions_shuffled = torch.tensor(context_positions)[context_perm]

                bins.append({
                    'bin_positions': bin_positions,
                    'context_positions_shuffled': context_positions_shuffled,
                    'target_token_ids': input_ids[b, bin_positions].cpu()
                })

            batch_bin_metadata.append({'valid_len': valid_len, 'bins': bins})

        # 3. 🚀 执行 Micro-Batch 推理
        all_sample_aucs = [[] for _ in range(batch_size)]

        max_bins_in_batch = 0
        for meta in batch_bin_metadata:
            if meta: max_bins_in_batch = max(max_bins_in_batch, len(meta['bins']))

        # 外层循环：Bin Index (这样可以保证负载均衡)
        for bin_idx in range(max_bins_in_batch):

            # --- 收集阶段：收集当前 Bin Index 下所有的任务 ---
            # 任务 = (样本ID, 步骤ID, Masked_Input)
            micro_batch_queue = []
            task_metadata_queue = []

            for sample_idx in range(batch_size):
                if batch_bin_metadata[sample_idx] and bin_idx < len(batch_bin_metadata[sample_idx]['bins']):
                    bin_data = batch_bin_metadata[sample_idx]['bins'][bin_idx]

                    ctx_shuffled = bin_data['context_positions_shuffled']  # CPU
                    bin_pos = bin_data['bin_positions']

                    # 为每个 Step 生成 Masked Input
                    for step_idx, ratio in enumerate(self.context_mask_ratios):
                        masked_ids = input_ids[sample_idx].cpu().clone()

                        # Mask Targets
                        masked_ids[bin_pos] = mask_id
                        # Mask Context
                        num_ctx_mask = int(np.round(ratio * len(ctx_shuffled)))
                        if num_ctx_mask > 0:
                            masked_ids[ctx_shuffled[:num_ctx_mask]] = mask_id

                        micro_batch_queue.append(masked_ids)

                        task_metadata_queue.append({
                            'sample_idx': sample_idx,
                            'step_idx': step_idx,
                            'bin_pos': bin_pos,
                            'target_ids': bin_data['target_token_ids']
                        })

            if not micro_batch_queue:
                continue

            # --- 执行阶段：按 MICRO_BATCH_SIZE 切片执行 ---
            total_tasks = len(micro_batch_queue)
            temp_results = {}  # key=(sample_idx, pos), val=[probs...]

            # 进度条可选
            # for i in range(0, total_tasks, MICRO_BATCH_SIZE):

            for i in range(0, total_tasks, MICRO_BATCH_SIZE):
                end_i = min(i + MICRO_BATCH_SIZE, total_tasks)

                # 1. 堆叠 Micro-Batch
                batch_inputs_cpu = torch.stack(micro_batch_queue[i:end_i])
                batch_metadata = task_metadata_queue[i:end_i]

                # 2. 移至 GPU (仅移动当前小批次)
                current_sample_indices = [meta['sample_idx'] for meta in batch_metadata]
                batch_att_mask = attention_mask[current_sample_indices]  # GPU Slice
                batch_inputs = batch_inputs_cpu.to(device)

                # 3. 前向传播 (混合精度以节省显存)
                try:
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(
                                input_ids=batch_inputs,
                                attention_mask=batch_att_mask if not shift_logits else None
                            )
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                            if shift_logits:
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                            logits = logits.float()
                            probs = F.softmax(logits, dim=-1)  # [Micro_Batch, Seq, Vocab]

                    # 4. 立即提取所需数据，释放 Logits 显存
                    for local_idx, meta in enumerate(batch_metadata):
                        sample_idx = meta['sample_idx']
                        step_idx = meta['step_idx']
                        bin_pos = meta['bin_pos']
                        target_ids = meta['target_ids']  # CPU

                        # 仅提取目标位置的概率
                        row_probs = probs[local_idx]  # GPU Slice

                        for pos, tid in zip(bin_pos, target_ids):
                            val = row_probs[pos, tid].item()  # Move scalar to CPU
                            key = (sample_idx, pos)

                            if key not in temp_results:
                                temp_results[key] = [0.0] * self.num_steps
                            temp_results[key][step_idx] = val

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # 最后的防线：如果 Micro-Batch=8 还挂，尝试清除缓存并报错提示
                        torch.cuda.empty_cache()
                        logging.error(f"OOM with MICRO_BATCH_SIZE={MICRO_BATCH_SIZE}. Please reduce it in the code.")
                        raise e
                    raise e

                # 清理当前 Micro-Batch 显存
                del batch_inputs, logits, probs, outputs

            # --- 聚合当前 Bin 的结果 ---
            for (s_idx, _), traj in temp_results.items():
                auc = self._calculate_trajectory_auc(np.array(traj))
                all_sample_aucs[s_idx].append(auc)

            # 清理 Bin 级缓存
            del micro_batch_queue, task_metadata_queue, temp_results
            torch.cuda.empty_cache()

        # 格式化输出
        results = []
        for b in range(batch_size):
            if batch_bin_metadata[b] is None or not all_sample_aucs[b]:
                results.append(None)
            else:
                results.append(all_sample_aucs[b])

        return results

    def _compute_bin_sequential(self, input_ids, attention_mask, bin_data,
                                model, mask_id, shift_logits, device):
        """Fallback sequential processing for OOM cases."""
        bin_positions = bin_data['bin_positions']
        target_token_ids = bin_data['target_token_ids']
        context_positions_shuffled = bin_data['context_positions_shuffled']

        trajectories = [[] for _ in range(len(bin_positions))]

        for context_mask_ratio in self.context_mask_ratios:
            num_context_to_mask = int(np.round(context_mask_ratio * len(context_positions_shuffled)))

            masked_ids = input_ids.clone()
            for pos in bin_positions:
                masked_ids[pos] = mask_id

            if num_context_to_mask > 0:
                positions_to_mask = context_positions_shuffled[:num_context_to_mask]
                masked_ids[positions_to_mask] = mask_id

            with torch.no_grad():
                outputs = model(
                    input_ids=masked_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0) if not shift_logits else None
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                if shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                probs = F.softmax(logits[0], dim=-1)

                for i, (pos, token_id) in enumerate(zip(bin_positions, target_token_ids)):
                    token_prob = probs[pos, token_id].item()
                    trajectories[i].append(token_prob)

        bin_aucs = []
        for traj in trajectories:
            if len(traj) > 1:
                auc = self._calculate_trajectory_auc(np.array(traj))
                bin_aucs.append(auc)

        return bin_aucs

    def _calculate_trajectory_auc(self, trajectory):
        """
        Calculate AUC of token probability trajectory.

        Correct Understanding:
        - X-axis: Context mask ratio (1.0 = fully masked → 0.0 = fully visible)
        - Y-axis: Token probability
        - Members: Probability rises earlier → Higher AUC
        - Non-members: Probability flat until late → Lower AUC
        """
        if len(trajectory) < 2:
            return 0.0

        # 🔥 KEY FIX: Use context_mask_ratios as X-axis
        x_values = self.context_mask_ratios

        auc = np.trapz(trajectory, x=x_values)
        x_range = x_values[-1] - x_values[0]  # Will be negative

        normalized_auc = auc / x_range if abs(x_range) > 1e-8 else auc

        return float(normalized_auc)

    def _aggregate_scores(self, scores):
        """Aggregate token-level scores using specified method."""
        if not scores:
            return 0.0

        arr = np.array(scores)

        if self.aggregation == 'mean':
            return float(np.mean(arr))
        elif self.aggregation == 'max':
            return float(np.max(arr))
        elif self.aggregation == 'min':
            return float(np.min(arr))
        elif self.aggregation == 'median':
            return float(np.median(arr))
        elif self.aggregation == 'top_k':
            k = max(1, int(len(arr) * self.aggregation_ratio))
            top_k_vals = np.partition(arr, -k)[-k:]
            return float(np.mean(top_k_vals))
        elif self.aggregation == 'bottom_k':
            k = max(1, int(len(arr) * self.aggregation_ratio))
            bottom_k_vals = np.partition(arr, k - 1)[:k]
            return float(np.mean(bottom_k_vals))
        else:
            return float(np.mean(arr))