# 主训练脚本：完成 memmap 数据加载、训练/验证、日志打印、checkpoint 保存和恢复。
import numpy as np
import torch
import argparse
import os
from typing import BinaryIO, IO

from .fullLM import TransformerLM
from .utils import AdamW, cross_entropy, gradient_clipping, get_lr_cosine_schedule

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 可以选择的起始位置范围是 [0, len(dataset) - context_length)
    start_indices = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )

    inputs = []
    targets = []

    for start in start_indices:
        input_sequence = dataset[start : start + context_length]
        target_sequence = dataset[start + 1 : start + context_length + 1]

        inputs.append(input_sequence)
        targets.append(target_sequence)

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    if isinstance(out, (str, os.PathLike)):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    """Estimate the average loss without computing gradients."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            total_loss += cross_entropy(model(inputs), targets).item()
    model.train()
    return total_loss / num_batches


def train(args: argparse.Namespace) -> None:
    """Train a TransformerLM using token IDs stored in binary files."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = np.dtype(args.data_dtype)
    train_data = np.memmap(args.train_bin, mode="r", dtype=dtype)
    valid_data = None
    if args.val_bin is not None:
        valid_data = np.memmap(args.val_bin, mode="r", dtype=dtype)

    d_ff = args.d_ff if args.d_ff is not None else int(8 * args.d_model / 3)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)

    for step in range(start_step, args.max_iters):
        learning_rate = get_lr_cosine_schedule(
            step,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        completed_step = step + 1
        if completed_step % args.log_interval == 0 or completed_step == 1:
            message = f"step {completed_step:6d} | train loss {loss.item():.4f}"
            if valid_data is not None:
                valid_loss = evaluate(
                    model,
                    valid_data,
                    args.batch_size,
                    args.context_length,
                    args.device,
                    args.eval_batches,
                )
                message += f" | val loss {valid_loss:.4f}"
            print(message, flush=True)

        if args.checkpoint_path and completed_step % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, completed_step, args.checkpoint_path)
            print(f"saved checkpoint to {args.checkpoint_path}", flush=True)

    if args.checkpoint_path:
        save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CS336 Transformer language model")
    parser.add_argument("--train-bin", required=True, help="Binary training token file")
    parser.add_argument("--val-bin", default=None, help="Optional binary validation token file")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=None)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoint-path", default="checkpoint.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--data-dtype", choices=("uint16", "int32", "int64"), default="uint16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # "gpu" is a convenient alias, but PyTorch uses "cuda" as the device name.
    if args.device == "gpu":
        args.device = "cuda"
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters
    return args


if __name__ == "__main__":
    train(parse_args())
