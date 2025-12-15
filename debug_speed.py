import torch
import time
import sys

sys.path.insert(0, '/home1/yibiao/code/DLM-MIA')

from attack.misc.models import ModelManager

model_path = "/home1/yibiao/code/DLM-MIA/checkpoints/LLaDA-8B-Base-pretrained-pretraining-512-mimir-arxiv_32_5.0e-5_4epoch"

# 🔥 修复：CUDA_VISIBLE_DEVICES=2 后，可见GPU编号变为0
device = torch.device("cuda:0")  # 改为 cuda:0

print("Loading model...")
model, tokenizer, _ = ModelManager.init_model(model_path, model_path, device)

print("Model loaded.  Starting speed test...")

# 准备输入
text = "This is a test sentence.  " * 50  # 约100 tokens
encoded = tokenizer.encode_plus(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

# 预热
print("Warming up...")
for _ in range(5):
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

# 测试
print("Testing forward pass speed...")
times = []
for i in range(20):
    start = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Pass {i + 1}: {elapsed:.4f}s")

avg_time = sum(times) / len(times)
print(f"\nAverage forward time: {avg_time:.4f}s")
print(f"Expected time for 18 calls: {18 * avg_time:.2f}s")
print(f"Expected time per sample (with overhead): {18 * avg_time * 1.2:.2f}s")

# 额外测试：不同序列长度
print("\n" + "=" * 50)
print("Testing different sequence lengths...")
for seq_len in [64, 128, 256, 512]:
    test_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    test_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    # 预热
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids=test_ids, attention_mask=test_mask)

    # 测试
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(input_ids=test_ids, attention_mask=test_mask)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print(f"Seq len {seq_len}: {avg:.4f}s")