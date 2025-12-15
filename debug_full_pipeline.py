import torch
import time
import sys
import random
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, '/home1/yibiao/code/DLM-MIA')

from attack.misc.models import ModelManager
from attack.attacks.utils import get_model_nll_params

model_path = "/home1/yibiao/code/DLM-MIA/checkpoints/LLaDA-8B-Base-pretrained-pretraining-512-mimir-arxiv_32_5.0e-5_4epoch"
device = torch.device("cuda:0")

print("Loading model...")
model, tokenizer, _ = ModelManager.init_model(model_path, model_path, device)
mask_id, shift_logits = get_model_nll_params(model)

print(f"Model loaded. mask_id={mask_id}, shift_logits={shift_logits}")

# 模拟真实文本
text = """
This is a sample academic paper abstract for testing purposes. 
We present a novel approach to membership inference attacks on diffusion language models.
Our method analyzes token-level probability trajectories during progressive context unmasking.
""" * 3  # 重复以接近真实长度

# 参数
num_steps = 6
num_bins = 3
max_length = 512
seed = 42
context_mask_ratios = np.linspace(1.0, 0.0, num_steps)

print(f"\nSimulating real attack with:")
print(f"  num_steps:  {num_steps}")
print(f"  num_bins: {num_bins}")
print(f"  text length: {len(text)} chars")

# ==================== 完整流程计时 ====================
total_start = time.time()

# Step 1: Tokenization
t1 = time.time()
encoded = tokenizer.encode_plus(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=max_length
)
input_ids = encoded["input_ids"][0].to(device)
attention_mask = encoded["attention_mask"][0].bool().to(device)
valid_len = int(attention_mask.sum().item())
tokenize_time = time.time() - t1
print(f"\n1. Tokenization: {tokenize_time:.4f}s (valid_len={valid_len})")

# Step 2: Bin setup
t2 = time.time()
all_valid_positions = list(range(1, valid_len - 1))
random.seed(seed)
random.shuffle(all_valid_positions)

num_bins = min(num_bins, len(all_valid_positions))
bin_size = len(all_valid_positions) // num_bins
setup_time = time.time() - t2
print(f"2. Bin setup: {setup_time:.4f}s (bins={num_bins}, bin_size={bin_size})")

# Step 3: Process bins
bin_times = []
forward_times = []
total_forward_calls = 0

attention_mask_batch = attention_mask.unsqueeze(0)

for bin_idx in range(num_bins):
    bin_start = time.time()

    start_idx = bin_idx * bin_size
    if bin_idx == num_bins - 1:
        end_idx = len(all_valid_positions)
    else:
        end_idx = (bin_idx + 1) * bin_size

    bin_positions = all_valid_positions[start_idx:end_idx]
    target_indices = torch.tensor(bin_positions, device=device)
    target_token_ids = input_ids[target_indices]

    context_positions = [p for p in range(valid_len) if p not in bin_positions]
    context_positions = torch.tensor(context_positions, device=device)

    g = torch.Generator(device=device)
    g.manual_seed(seed + bin_idx)
    context_perm = torch.randperm(len(context_positions), generator=g, device=device)
    context_positions_shuffled = context_positions[context_perm]

    bin_trajectories = [[] for _ in range(len(target_indices))]

    # Process steps for this bin
    for context_mask_ratio in context_mask_ratios:
        forward_start = time.time()

        num_context_to_mask = int(np.round(context_mask_ratio * len(context_positions)))

        masked_ids = input_ids.clone()
        masked_ids[target_indices] = mask_id

        if num_context_to_mask > 0:
            positions_to_mask = context_positions_shuffled[:num_context_to_mask]
            masked_ids[positions_to_mask] = mask_id

        with torch.no_grad():
            outputs = model(
                input_ids=masked_ids.unsqueeze(0),
                attention_mask=attention_mask_batch if not shift_logits else None
            )
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            if shift_logits:
                logits = torch.cat([logits[:, : 1, :], logits[:, :-1, :]], dim=1)

            probs = F.softmax(logits[0], dim=-1)

            for i, (pos, token_id) in enumerate(zip(target_indices, target_token_ids)):
                token_prob = probs[pos, token_id].item()
                bin_trajectories[i].append(token_prob)

        forward_time = time.time() - forward_start
        forward_times.append(forward_time)
        total_forward_calls += 1

    bin_time = time.time() - bin_start
    bin_times.append(bin_time)

    print(f"3. {bin_idx + 1}.  Bin {bin_idx}:  {bin_time:.4f}s ({len(bin_positions)} tokens, {num_steps} steps)")

total_time = time.time() - total_start

print(f"\n" + "=" * 60)
print(f"SUMMARY:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Tokenization: {tokenize_time:.4f}s ({tokenize_time / total_time * 100:.1f}%)")
print(f"  Bin setup: {setup_time:.4f}s ({setup_time / total_time * 100:.1f}%)")
print(f"  Bin processing: {sum(bin_times):.2f}s ({sum(bin_times) / total_time * 100:.1f}%)")
print(f"\n  Total forward calls: {total_forward_calls}")
print(f"  Avg forward time: {np.mean(forward_times):.4f}s")
print(f"  Min forward time: {np.min(forward_times):.4f}s")
print(f"  Max forward time: {np.max(forward_times):.4f}s")
print(f"  Total forward time: {sum(forward_times):.2f}s ({sum(forward_times) / total_time * 100:.1f}%)")
print(
    f"\n  Non-forward overhead: {total_time - sum(forward_times):.2f}s ({(total_time - sum(forward_times)) / total_time * 100:.1f}%)")
print(f"=" * 60)

print(f"\nExpected time for 500 samples: {total_time * 500 / 60:.1f} minutes")