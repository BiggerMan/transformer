"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 整个公式：FFN(x) = max(0, xW1 + b1)W2 + b2
        # 这就是一个简单的MLP： 输入层 → 隐藏层 → 输出层
        # 前馈神经网络（Feedforward Neural Network）：
        # • 单向传播：数据只向一个方向流动
        # • 全连接：相邻层之间神经元全连接
        # • 非线性：每层后使用激活函数（ReLU、sigmoid等）
        x = self.linear1(x)
        # ReLU函数的数学公式是：f(x) = max(0, x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
