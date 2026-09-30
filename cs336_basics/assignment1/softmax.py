# 注意力模块：实现数值稳定 Softmax、缩放点积注意力、多头因果注意力和 RoPE 注意力。
import torch
from torch import nn

from .linear import Linear
from .pre_norm_block import RotaryPositionalEmbedding


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shift = x - x_max
    x_exp = torch.exp(x_shift)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 对每个 query 在所有 key 上进行归一化。
    attention_weights = softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        batch_shape = x.shape[:-2]

        # 先对输入进行线性变换，得到 Q、K、V。
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        # 把 d_model 切分成 n_heads 份，再把 n_heads 移到序列维前面。
        # (..., seq_len, d_model) -> (..., n_heads, seq_len, head_dim)
        Q = Q.reshape(*batch_shape, seq_len, self.n_heads, self.head_dim).transpose(-3, -2)
        K = K.reshape(*batch_shape, seq_len, self.n_heads, self.head_dim).transpose(-3, -2)
        V = V.reshape(*batch_shape, seq_len, self.n_heads, self.head_dim).transpose(-3, -2)

        # 下三角为 True：当前位置只能看到自己以及之前的 token。
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attention = scaled_dot_product_attention(Q, K, V, causal_mask)

        # 把所有头重新拼接起来。
        # (..., n_heads, seq_len, head_dim) -> (..., seq_len, d_model)
        attention = attention.transpose(-3, -2)
        attention = attention.reshape(*batch_shape, seq_len, self.d_model)
        return self.wo(attention)


###这个多头自注意力也不是很懂
class CausalMultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        batch_shape = x.shape[:-2]

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        # 先拆分多个头，RoPE 作用的维度是每个头的 head_dim。
        Q = Q.reshape(*batch_shape, seq_len, self.n_heads, self.head_dim).transpose(-3, -2)
        K = K.reshape(*batch_shape, seq_len, self.n_heads, self.head_dim).transpose(-3, -2)
        V = V.reshape(*batch_shape, seq_len, self.n_heads, self.head_dim).transpose(-3, -2)

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # RoPE 只应用于 Q 和 K，不应用于 V。
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attention = scaled_dot_product_attention(Q, K, V, causal_mask)

        attention = attention.transpose(-3, -2)
        attention = attention.reshape(*batch_shape, seq_len, self.d_model)
        return self.wo(attention)

