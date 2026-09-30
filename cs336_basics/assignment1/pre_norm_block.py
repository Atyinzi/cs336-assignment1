# Transformer 基础模块：实现 RMSNorm、SwiGLU 和 Rotary Positional Embedding。
import torch
from torch import nn

from .linear import Linear

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  

        rms = torch.sqrt(
            torch.sum(x * x, dim=-1, keepdim=True) / self.d_model + self.eps
        )

        result = (x / rms) * self.g

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.w1(x)                    
        h3 = self.w3(x)                     
        silu_h1 = h1 * torch.sigmoid(h1)    
        gate = silu_h1 * h3                  
        return self.w2(gate)  

###这部分不是很理解，rope
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k should be even")

        self.theta = theta
        self.d_k = d_k  # RoPE 作用的向量维度；多头注意力中通常是每个 head 的维度。
        self.max_seq_len = max_seq_len

        # 每相邻两维共用一个频率，得到 d_k/2 个频率。
        frequencies = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(positions, frequencies)  # [max_seq_len, d_k/2]

        # 位置编码是固定值，不是可学习参数。
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # token_positions 可以只有序列维，也可以带 batch 维；补齐缺少的维度。
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # stack 将旋转后的偶数维、奇数维交错放回原顺序。
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.flatten(-2).to(x.dtype)



    
