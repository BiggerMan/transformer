"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        # 分析点1：
        # (src != self.src_pad_idx) 返回一个布尔类型的张量，用于注意力机制屏蔽填充位置。
        # 假设 src = [[5, 8, 3, 1, 1], [7, 2, 1, 1, 1]]， pad_idx 是 1 ，
        # (src != 1) = [[True, True, True, False, False],
        #               [True, True, False, False, False]]
        # 分析点2：
        # unsqueeze会在指定位置插入一个大小为1的新维度。
        # 解释：
        # 第1维(1)：自动广播到所有注意力头
        # 第2维(1)：自动广播到所有目标位置
        # 结果：每个注意力头、每个目标位置都使用相同的源序列掩码
        # 效果：
        # 原始掩码: [batch_size, src_len]  # 每个序列的掩码
        #        ↓ unsqueeze(1)
        #        [batch_size, 1, src_len]
        #        ↓ unsqueeze(2)
        #        [batch_size, 1, 1, src_len]  # 最终形状
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask