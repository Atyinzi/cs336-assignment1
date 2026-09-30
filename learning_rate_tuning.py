# 学习率实验脚本：用于比较不同学习率配置对训练过程的影响。
import torch

from cs336_basics.utils import SGD


# 三组实验使用完全相同的初始参数，方便公平比较。
torch.manual_seed(42)
initial_weights = 5 * torch.randn((10, 10))

for lr in [1e1, 1e2, 1e3]:
    weights = torch.nn.Parameter(initial_weights.clone())
    optimizer = SGD([weights], lr=lr)

    print(f"\nlr = {lr}")

    for step in range(10):
        # 清除上一次迭代留下的梯度。
        optimizer.zero_grad()

        # 构造一个简单的目标：让所有参数逐渐接近 0。
        loss = (weights**2).mean()

        print(f"step={step:2d}, loss={loss.item():.4f}")

        # 计算梯度。
        loss.backward()

        # 使用 SGD 更新参数。
        optimizer.step()
