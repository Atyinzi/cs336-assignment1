# MiniTrain: 从单卡 Transformer 到分布式训练系统

这是基于 Stanford CS336 Assignment 1/2 完成的教学型训练系统。项目从可读的
Transformer 训练循环出发，逐步加入混合精度、梯度累积、activation checkpointing、
SDPA/Triton attention、DDP 梯度分桶、ZeRO-1、FSDP、通信重叠、分片 checkpoint 和文本生成。

项目目标不是复刻 PyTorch/DeepSpeed 的全部生产能力，而是用较少代码讲清楚训练系统中的
三个问题：显存花在哪里、通信什么时候发生、优化为什么确实有效。

原始作业说明见 [cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)，逐题推导见
[WRITEUP_GUIDE_CN.md](./WRITEUP_GUIDE_CN.md)。

## 当前结果

单卡训练环境：NVIDIA GeForce RTX 3050 Laptop GPU，TinyStories，10K byte-level BPE，
约 128.6M 参数，context length 128，micro batch 1，gradient accumulation 4，BF16。

| 项目 | 结果 |
|---|---:|
| 完成训练步数 | 6000 |
| 5000-step loss EMA | 2.3881 |
| 6000-step loss EMA | 2.3163 |
| 固定 10 个验证片段：5000-step loss | 2.5538 |
| 固定 10 个验证片段：6000-step loss | 2.4901 |
| 5000→6000 相同配置验证 loss 改善 | 约 2.5% |
| 最后 100 步平均吞吐 | 约 1091 tokens/s |
| 训练峰值显存 | 约 2.97 GiB allocated |

当前推荐 checkpoint 是本地的 `results/minitrain_step6000.pt`。checkpoint 约 1.5GB，包含
FP32 参数与 AdamW 状态，因此不提交到 Git。

### Attention 后端实测

固定配置：small model、context 128、batch 1、BF16、5 次 warmup、20 次测量。

| Backend | Step time | Tokens/s | 相对 naive |
|---|---:|---:|---:|
| naive | 158.14 ms | 809.4 | 1.00x |
| Triton | 154.04 ms | 830.9 | 1.03x |
| SDPA | **139.89 ms** | **915.0** | **1.13x** |

因此本机默认选择 SDPA。当前自定义 Triton forward 正确，但 backward 仍使用 PyTorch
重计算；在短序列和小显卡上没有胜过 PyTorch SDPA，这也是一个真实的工程结论。

## 架构

```text
TinyStories text
      │
      ▼
10K byte-level BPE ──► uint16 memmap
      │
      ▼
TransformerLM (RMSNorm + RoPE + SwiGLU + causal attention)
      │
      ├── naive / SDPA / Triton attention
      ├── BF16 autocast
      ├── gradient accumulation
      ├── activation checkpointing
      └── cosine learning-rate schedule
      │
      ▼
DDP buckets / ZeRO-1 / educational FSDP
      │
      ▼
full model checkpoint + per-rank optimizer shards
      │
      ▼
validation loss + temperature/top-p generation
```

主要文件：

| 文件 | 作用 |
|---|---|
| `scripts/minitrain.py` | 数据、训练、验证、日志、checkpoint、恢复与并行策略入口 |
| `scripts/minitrain_inference.py` | 从 checkpoint 评估验证 loss 或生成文本 |
| `cs336_systems/attention.py` | tiled PyTorch attention 与 Triton kernel |
| `cs336_systems/distributed.py` | bucketed DDP、ZeRO-1、FSDP 与通信重叠 |
| `cs336_systems/tokenizer.py` | 加载 Assignment 1 byte-level BPE tokenizer |
| `scripts/benchmark_transformer.py` | 单卡 forward/backward/optimizer benchmark |
| `scripts/benchmark_parallel.py` | 单种多 GPU 策略 benchmark |
| `scripts/run_parallel_suite.py` | 自动运行并汇总全部并行策略 |

## 快速开始

### 安装与测试

```bash
cd assignment2-systems
uv sync
uv run pytest -q
```

在仓库根目录准备 TinyStories 原始文本后，用已提交的 10K tokenizer 编码数据：

```bash
cd ..
uv run python -m cs336_basics.assignment1.scripts.encode_data \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories_train.bin \
  --tokenizer artifacts/tinystories_bpe_10000/tokenizer.pkl
uv run python -m cs336_basics.assignment1.scripts.encode_data \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/tinystories_valid.bin \
  --tokenizer artifacts/tinystories_bpe_10000/tokenizer.pkl
cd assignment2-systems
```

数据与模型 checkpoint 需要在本地生成；Git 仓库保存代码、tokenizer 和小型实验日志。

### 从头训练

```bash
uv run python scripts/minitrain.py \
  --train-bin ../data/tinystories_train.bin \
  --val-bin ../data/tinystories_valid.bin \
  --model-size small \
  --vocab-size 10000 \
  --context-length 128 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --steps 6000 \
  --eval-every 100 \
  --eval-steps 10 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 100 \
  --lr-decay-steps 10000 \
  --cosine-lr \
  --attention-backend sdpa \
  --mixed-precision \
  --output results/minitrain_step6000.pt \
  --log-file results/minitrain_step6000.jsonl \
  --device cuda
```

这是从头训练的参考命令；已有 6000 步结果来自先训练到 5000 步，再恢复 1000 步。
随机数消耗顺序、PyTorch 版本与运行环境可能导致重新运行的曲线略有差异。

### 从 6000-step checkpoint 生成文本

```bash
uv run python scripts/minitrain_inference.py \
  --checkpoint results/minitrain_step6000.pt \
  --tokenizer ../artifacts/tinystories_bpe_10000/tokenizer.pkl \
  --prompt "Once upon a time, there was a little dragon" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-p 0.95 \
  --attention-backend sdpa \
  --mixed-precision \
  --device cuda
```

一次真实输出：

> Once upon a time, there was a little girl named Amy. Amy loved to play in the yard. One day,
> she found a big, icy tree in the yard...

### 继续训练

```bash
uv run python scripts/minitrain.py \
  --train-bin ../data/tinystories_train.bin \
  --val-bin ../data/tinystories_valid.bin \
  --model-size small \
  --vocab-size 10000 \
  --context-length 128 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --steps 1000 \
  --eval-every 100 \
  --eval-steps 10 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 100 \
  --lr-decay-steps 10000 \
  --cosine-lr \
  --attention-backend sdpa \
  --mixed-precision \
  --resume results/minitrain_step6000.pt \
  --output results/minitrain_step7000.pt \
  --log-file results/minitrain_step7000.jsonl \
  --device cuda
```

继续训练不是项目完成的必要条件。应根据固定验证集 loss 决定是否继续，而不是只看训练步数。

## 分布式策略

### DDP + ZeRO-1

```bash
uv run torchrun --standalone --nproc-per-node=2 scripts/minitrain.py \
  --parallel-strategy ddp \
  --zero-stage 1 \
  --ddp-bucket-cap-mb 25 \
  ...
```

DDP 同步分桶后的平均梯度；ZeRO-1 只让每个 rank 保存和更新一部分 AdamW 状态，更新后再
广播参数。

### FSDP

```bash
uv run torchrun --standalone --nproc-per-node=2 scripts/minitrain.py \
  --parallel-strategy fsdp \
  --fsdp-prefetch \
  ...
```

教学版 FSDP 对 Linear/Embedding 权重分片，前向/反向按需 All-Gather，梯度通过
Reduce-Scatter 回到本地分片，并异步预取下一层参数。FSDP 已经包含优化器状态分片，不能再
同时指定 `--zero-stage 1`。

### 自动并行实验

在至少两张 GPU 的机器上：

```bash
uv run python scripts/run_parallel_suite.py \
  --nproc-per-node 2 \
  --model-size small \
  --context-length 128 \
  --batch-size 2 \
  --warmup 5 \
  --steps 20
```

它会依次比较逐参数 DDP、flat DDP、bucketed DDP、ZeRO-1、FSDP 与 FSDP prefetch，输出
step time、tokens/s、暴露通信时间、峰值显存和每 rank 状态大小，并生成 Markdown 报告。

## 证据边界

以下结论经过本机实测：

- RTX 3050 上的 6000-step TinyStories 训练、断点恢复、验证和生成；
- naive、SDPA、Triton attention 的单卡性能；
- Triton attention forward/backward 数值测试；
- CPU/Gloo 两进程下 DDP、共享权重、ZeRO-1、FSDP FP32/FP16、checkpoint round-trip。

以下内容已经实现，但由于本机只有一张 GPU，没有声称多 GPU 性能收益：

- NCCL DDP bucket 性能；
- ZeRO-1 显存缩减比例；
- FSDP 与 prefetch 的多 GPU 吞吐和显存数据。

当前 FSDP 是教学实现，不包含生产 PyTorch FSDP 的全部能力，例如完整的 nested wrapping、
弹性 world-size checkpoint 重分片和成熟的通信调度。Triton backward 也尚未实现为完全融合
kernel。这些限制会在简历和面试表述中明确保留。
