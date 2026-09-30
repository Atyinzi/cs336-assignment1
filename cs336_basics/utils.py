# 训练工具：实现交叉熵、SGD、AdamW、梯度裁剪和 cosine 学习率调度。
import math
from collections.abc import Callable, Iterable

import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (..., vocab_size)
    targets: (...) 整数张量，取值在 [0, vocab_size)
    返回标量：所有位置的平均交叉熵
    """
    # 1. 减去最大值，防止 exp 溢出
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits

    # 2. 计算 logsumexp = max + log(sum(exp(shifted)))
    logsumexp = max_logits.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))

    # 3. 取出目标 token 对应的 logit
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ##logits = [[0.5, 1.2, 0.3],
            #[0.8, 0.1, 0.9]]
    ##targets = [1, 2]
    # targets.unsqueeze(-1) = [[1], [2]]
    # gather 结果 = [[1.2], [0.9]]
    # squeeze 后 = [1.2, 0.9]

    # 4. 交叉熵 = logsumexp - target_logit
    loss = logsumexp - target_logits

    # 5. 对所有前导维度求平均
    return loss.mean()

class SGD(torch.optim.Optimizer):
    """使用 1 / sqrt(t + 1) 衰减学习率的 SGD。"""

    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * p.grad.data
                state["t"] = t + 1

        return loss


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip all parameter gradients together so their L2 norm is at most max_l2_norm."""
    parameters = list(parameters)  # An iterator can only be traversed once.
    gradients = [p.grad for p in parameters if p.grad is not None]
    if len(gradients) == 0:
        return

    # The norm of the full gradient vector is sqrt(sum of all squared entries).
    total_norm = torch.sqrt(sum(torch.sum(gradient * gradient) for gradient in gradients))
    scale = max_l2_norm / (total_norm + 1e-6)

    if scale < 1:
        for gradient in gradients:
            gradient.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate at iteration it."""
    if it < warmup_iters:
        # Linear warmup: start at 0 and reach max_learning_rate.
        return max_learning_rate * it / warmup_iters

    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_value = math.cos(math.pi * progress)
        return min_learning_rate + 0.5 * (1 + cosine_value) * (max_learning_rate - min_learning_rate)

    return min_learning_rate


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decoupled weight decay uses the unadjusted learning rate.
                parameter.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                adjusted_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-adjusted_lr)

        return loss
