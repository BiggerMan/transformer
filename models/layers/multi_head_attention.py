"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
import os
import torch
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # nn.Linear 是 PyTorch 中的全连接层（线性变换层），作用是对输入进行线性变换
        # y = xW^T + b
        #   • x 是输入向量
        #   • W 是权重矩阵
        #   • b 是偏置向量
        #   • y 是输出向量
        # ^T 表示转置操作。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # 将注意力矩阵保存到一个文件里，之后用matplotlib，做它的变化过程的可视化。
        # 保存attention矩阵到文件
        if not os.path.exists('result'):
            os.makedirs('result')
        filename = "result/tensor_changes.pt"
        torch.save(attention, filename)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        # 通过transpose交换张量的两个维度，
        # 目的：将注意力头维度移到前面，便于后续矩阵乘法。使q @ k_t计算时维度对齐。
        # 效果如下：
        # 原始: [batch, seq_len, n_head, d_k]
        #        ↓ transpose(1, 2)
        #     结果: [batch, n_head, seq_len, d_k]
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        # transpose(1, 2): 将head维度移回原始位置
        # contiguous(): 确保内存连续，便于view操作
        # view(): 重塑张量形状，合并多头结果
        # 维度变化：
        # 输入: [batch_size, head, length, d_tensor]
        #        ↓ transpose(1, 2)
        #        [batch_size, length, head, d_tensor]
        #        ↓ contiguous().view()
        # 输出: [batch_size, length, d_model]  # d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
