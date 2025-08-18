"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

# LayerNorm（层归一化）是对神经网络中单个样本的所有特征（理解为一行）进行归一化的技术
# output = (x - mean) / sqrt(var + eps) * γ + β
# 位置：每个子层（注意力、前馈网络）之后
# 效果：防止梯度消失/爆炸，提高训练稳定性
# 与之相对的是，BatchNorm（批量归一化）是对神经网络中所有特征（理解为一列）进行归一化的技术。
# 与BatchNorm区别：不依赖batch size（因为Size可能会变），适合序列模型。

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
