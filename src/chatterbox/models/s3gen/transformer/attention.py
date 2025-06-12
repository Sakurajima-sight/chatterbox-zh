# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-Head Attention 层的定义"""

import math
from typing import Tuple

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """多头注意力层。
    
    参数:
        n_head (int): 注意力头的数量。
        n_feat (int): 输入的特征数量。
        dropout_rate (float): Dropout 比例。
        key_bias (bool): 是否在键（key）上加偏置（默认为 True）。

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        """构造 MultiHeadedAttention 对象。

        参数：
            - n_head: 注意力头的数量
            - n_feat: 输入特征的维度
            - dropout_rate: Dropout 的比例
            - key_bias: 是否为键添加偏置，默认为 True
        """
        super().__init__()
        assert n_feat % n_head == 0  # 确保特征维度可以被头的数量整除
        # 假设 d_v 等于 d_k，即键和值的维度相同
        self.d_k = n_feat // n_head  # 每个头的键和值的维度
        self.h = n_head  # 注意力头的数量

        # 定义查询、键、值的线性变换
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        
        # 定义 dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将查询、键和值进行线性变换。

        参数:
            - query (torch.Tensor): 查询张量，形状为 (#batch, time1, size)
            - key (torch.Tensor): 键张量，形状为 (#batch, time2, size)
            - value (torch.Tensor): 值张量，形状为 (#batch, time2, size)

        返回：
            - query (torch.Tensor): 转换后的查询张量，形状为 (#batch, n_head, time1, d_k)
            - key (torch.Tensor): 转换后的键张量，形状为 (#batch, n_head, time2, d_k)
            - value (torch.Tensor): 转换后的值张量，形状为 (#batch, n_head, time2, d_k)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        # 调整维度顺序，变为 (batch, head, time1, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """计算注意力上下文向量。

        参数:
            - value (torch.Tensor): 转换后的值张量，形状为 (#batch, n_head, time2, d_k)
            - scores (torch.Tensor): 注意力分数，形状为 (#batch, n_head, time1, time2)
            - mask (torch.Tensor): 掩码张量，形状为 (#batch, 1, time2) 或 (#batch, time1, time2)

        返回：
            - 输出张量，形状为 (#batch, time1, d_model)，根据注意力分数加权的值张量。
        """
        n_batch = value.size(0)
        
        # 如果提供了 mask，则根据 mask 对注意力分数进行掩码处理
        if mask.size(2) > 0:  # 当 time2 > 0 时
            mask = mask.unsqueeze(1).eq(0)  # 扩展 mask，形状为 (batch, 1, *, time2)
            mask = mask[:, :, :, :scores.size(-1)]  # 对 scores 长度进行修正
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)  # 计算 softmax 注意力分数

        # 进行 Dropout 操作
        p_attn = self.dropout(attn)
        
        # 用加权的注意力分数计算值张量
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))  # (batch, time1, d_model)

        # 返回经过线性变换后的结果
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算缩放点积注意力。

        参数：
            - query (torch.Tensor): 查询张量，形状为 (#batch, time1, size)
            - key (torch.Tensor): 键张量，形状为 (#batch, time2, size)
            - value (torch.Tensor): 值张量，形状为 (#batch, time2, size)
            - mask (torch.Tensor): 掩码张量，形状为 (#batch, time1, time2)
            - cache (torch.Tensor): 缓存张量，形状为 (1, head, cache_t, d_k * 2)

        返回：
            - 输出张量：形状为 (#batch, time1, d_model)
            - 缓存张量：形状为 (1, head, cache_t + time1, d_k * 2)
        """
        q, k, v = self.forward_qkv(query, key, value)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        
        # 将 k 和 v 拼接为新的缓存
        new_cache = torch.cat((k, v), dim=-1)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 使用加权的注意力分数计算输出
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """具有相对位置编码的多头注意力层。
    参考论文: https://arxiv.org/abs/1901.02860

    参数:
        n_head (int): 注意力头的数量。
        n_feat (int): 特征的数量。
        dropout_rate (float): Dropout 比例。
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        """构建 RelPositionMultiHeadedAttention 对象。

        参数：
            - n_head: 注意力头的数量
            - n_feat: 特征维度
            - dropout_rate: Dropout 比例
            - key_bias: 是否在键（Key）中加入偏置，默认为 True
        """
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        
        # 定义用于位置编码的线性变换
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        
        # 用于矩阵 c 和矩阵 d 的两个可学习的偏置参数，参见论文 https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        
        # 使用 Xavier 均匀初始化偏置
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """计算相对位置编码。

        参数：
            x (torch.Tensor): 输入张量，形状为 (batch, head, time1, 2*time1-1)，
                time1 表示查询向量的长度。

        返回：
            torch.Tensor: 输出张量，形状为 (batch, head, time1, time2)。
        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # 只保留从 0 到 time2 的位置
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算具有相对位置编码的缩放点积注意力。

        参数：
            query (torch.Tensor): 查询张量，形状为 (#batch, time1, size)
            key (torch.Tensor): 键张量，形状为 (#batch, time2, size)
            value (torch.Tensor): 值张量，形状为 (#batch, time2, size)
            mask (torch.Tensor): 掩码张量，形状为 (#batch, 1, time2) 或
                (#batch, time1, time2)，(0, 0, 0) 表示假掩码
            pos_emb (torch.Tensor): 位置嵌入张量，形状为 (#batch, time2, size)
            cache (torch.Tensor): 缓存张量，形状为 (1, head, cache_t, d_k * 2)
                其中 `cache_t == chunk_size * num_decoding_left_chunks`
                和 `head * d_k == size`

        返回：
            torch.Tensor: 输出张量，形状为 (#batch, time1, d_model)
            torch.Tensor: 缓存张量，形状为 (1, head, cache_t + time1, d_k * 2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # 对于缓存数据，如果存在则拼接
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        
        # 拼接新的缓存
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # 在查询中加入相对位置偏置
        q_with_bias_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2)

        # 计算注意力分数
        # 计算矩阵 a 和矩阵 c，如论文中所述 https://arxiv.org/abs/1901.02860 Section 3.3
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # 计算矩阵 b 和矩阵 d
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        
        # 处理相对位置的平移
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        # 计算最终的注意力分数
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        # 返回经过注意力加权的输出和新的缓存
        return self.forward_attention(v, scores, mask), new_cache
