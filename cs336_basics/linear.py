# 基础神经网络层：实现 Linear 和 Embedding，并完成参数初始化与前向计算。
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self,in_features: int,out_features: int,device:torch.device |None=None, dtype:torch.dtype| None=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W,std=std,a=-3*std,b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x@ self.W.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim 
        self.device = device
        self.dtype = dtype

        self.embedding_martix = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))
        std=1
        torch.nn.init.trunc_normal_(self.embedding_martix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_martix[token_ids]


