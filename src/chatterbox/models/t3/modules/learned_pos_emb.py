from typing import Union

import torch
from torch import nn, Tensor


class LearnedPositionEmbeddings(nn.Module):
    """
    该类实现了位置嵌入（Positional Embeddings），用于在序列中为每个位置分配一个向量表示。
    这种位置嵌入通常用于Transformer模型，以便模型能够区分输入序列中不同位置的元素。
    """

    def __init__(self, seq_len, model_dim, init=.02):
        """
        初始化位置嵌入。

        :param seq_len: 序列的最大长度，即模型能够处理的最大输入长度
        :param model_dim: 嵌入的维度（即模型的隐藏层维度）
        :param init: 嵌入权重初始化的标准差，默认值为0.02（与GPT-2的标准初始化一致）
        """
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # 使用正态分布初始化位置嵌入权重
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        """
        前向传播：根据输入的序列x的长度返回对应的位置信息嵌入。

        :param x: 输入序列，形状为(B, T)，其中B是批量大小，T是序列长度
        :return: 位置嵌入，形状为(T, dim)，即返回的嵌入与输入序列长度相匹配
        """
        sl = x.shape[1]  # 获取输入序列的长度
        return self.emb(torch.arange(0, sl, device=x.device))  # 返回对应长度的位置信息嵌入

    def get_fixed_embedding(self, idx: 'Union[int, Tensor]'):
        """
        获取指定位置的固定位置嵌入，支持单个索引或批量索引。

        :param idx: 位置索引，可以是单个整数、形状为(T,)的整数张量，或者形状为(B, T)的二维整数张量
        :return: 返回给定索引位置的嵌入，形状为(B, T, dim)，即对于单一整数输入返回形状为(1, 1, dim)
        """
        device = self.emb.weight.device  # 获取位置嵌入权重所在的设备
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)  # 确保idx在正确的设备上
        idx = torch.atleast_2d(idx)  # 将idx转换为至少二维张量
        assert idx.ndim == 2  # 确保idx的维度为2
        return self.emb(idx)  # 返回对应的位置信息嵌入，形状为(B, T, dim)
