# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn

class TransformerEncoderLayer(nn.Module):
    """
    编码器层模块，包含自注意力和前馈网络模块。

    :param size: 输入维度。
    :param self_attn: 自注意力模块实例。
        可以使用 `MultiHeadedAttention` 或 `RelPositionMultiHeadedAttention`。
    :param feed_forward: 前馈网络模块实例。
        可以使用 `PositionwiseFeedForward`。
    :param dropout_rate: dropout 概率。
    :param normalize_before: 
        - True: 在每个子块之前使用 LayerNorm。
        - False: 在每个子块之后使用 LayerNorm。
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """
        构造 EncoderLayer 对象。

        :param size: 输入维度。
        :param self_attn: 自注意力模块。
        :param feed_forward: 前馈网络模块。
        :param dropout_rate: dropout 的概率。
        :param normalize_before: 是否在子块之前进行LayerNorm。
        """
        super().__init__()
        self.self_attn = self_attn  # 自注意力模块
        self.feed_forward = feed_forward  # 前馈网络模块
        self.norm1 = nn.LayerNorm(size, eps=1e-12)  # 第一层的LayerNorm
        self.norm2 = nn.LayerNorm(size, eps=1e-12)  # 第二层的LayerNorm
        self.dropout = nn.Dropout(dropout_rate)  # dropout层
        self.size = size
        self.normalize_before = normalize_before  # 是否在子模块之前进行归一化

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算编码后的特征。

        :param x: 输入张量，形状为 (#batch, time, size)
        :param mask: 输入的 mask 张量，形状为 (#batch, time, time)，
            (0, 0, 0) 表示伪 mask。
        :param pos_emb: 位置嵌入张量，用于与 ConformerEncoderLayer 兼容。
        :param mask_pad: 用于统一API与Conformer接口，不在transformer层使用。
        :param att_cache: 自注意力的缓存张量，
            形状为 (#batch=1, head, cache_t1, d_k * 2)，head * d_k == size。
        :param cnn_cache: Conformer层的卷积缓存，
            形状为 (#batch=1, size, cache_t2)，在这里不使用，仅用于与Conformer接口兼容。

        :return:
            - 输出张量 (#batch, time, size)
            - mask 张量 (#batch, time, time)
            - 更新后的注意力缓存张量 (#batch=1, head, cache_t1 + time, d_k * 2)
            - 卷积缓存张量 (#batch=1, size, cache_t2)
        """
        residual = x  # 保存输入张量作为残差
        if self.normalize_before:  # 如果在子模块前进行归一化
            x = self.norm1(x)
        # 计算自注意力，并更新缓存
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb=pos_emb, cache=att_cache)
        # 将自注意力的输出与输入进行跳跃连接，并应用dropout
        x = residual + self.dropout(x_att)
        if not self.normalize_before:  # 如果不在子模块前进行归一化
            x = self.norm1(x)

        residual = x  # 保存跳跃连接后的张量作为残差
        if self.normalize_before:  # 如果在子模块前进行归一化
            x = self.norm2(x)
        # 计算前馈网络的输出，并应用dropout
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:  # 如果不在子模块前进行归一化
            x = self.norm2(x)

        # 伪卷积缓存（在Transformer中不使用）
        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache  # 返回处理后的张量和相关缓存


class ConformerEncoderLayer(nn.Module):
    """
    编码器层模块，包含自注意力、前馈网络、卷积模块等，适用于Conformer模型。

    :param size: 输入维度。
    :param self_attn: 自注意力模块实例，支持 `MultiHeadedAttention` 或 `RelPositionMultiHeadedAttention`。
    :param feed_forward: 前馈网络模块实例，支持 `PositionwiseFeedForward`。
    :param feed_forward_macaron: 额外的前馈网络模块实例，支持 `PositionwiseFeedForward`。
    :param conv_module: 卷积模块实例，支持 `ConvlutionModule`。
    :param dropout_rate: dropout 概率。
    :param normalize_before: 
        - True: 在每个子模块之前使用 LayerNorm。
        - False: 在每个子模块之后使用 LayerNorm。
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """
        构造一个 EncoderLayer 对象。

        :param size: 输入维度。
        :param self_attn: 自注意力模块。
        :param feed_forward: 前馈网络模块。
        :param feed_forward_macaron: 额外的前馈网络模块。
        :param conv_module: 卷积模块。
        :param dropout_rate: dropout 的概率。
        :param normalize_before: 是否在每个子模块之前使用 LayerNorm。
        """
        super().__init__()
        self.self_attn = self_attn  # 自注意力模块
        self.feed_forward = feed_forward  # 前馈网络模块
        self.feed_forward_macaron = feed_forward_macaron  # 可选的额外前馈模块
        self.conv_module = conv_module  # 卷积模块
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # 用于前馈模块的 LayerNorm
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # 用于多头自注意力模块的 LayerNorm
        
        # 如果有额外的前馈网络模块，则使用一个额外的 LayerNorm
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5  # macaron风格的前馈模块缩放因子
        else:
            self.ff_scale = 1.0
        
        # 如果有卷积模块，则需要卷积模块的 LayerNorm
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)  # 用于卷积模块的 LayerNorm
            self.norm_final = nn.LayerNorm(size, eps=1e-12)  # 用于最终输出的 LayerNorm
        
        self.dropout = nn.Dropout(dropout_rate)  # dropout层
        self.size = size  # 输入维度
        self.normalize_before = normalize_before  # 是否在子模块前进行归一化

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算编码后的特征。

        :param x: 输入张量，形状为 (#batch, time, size)
        :param mask: 输入的 mask 张量，形状为 (#batch, time, time)，
            (0, 0, 0) 表示伪 mask。
        :param pos_emb: 位置嵌入，ConformerEncoderLayer 中必须提供。
        :param mask_pad: 用于卷积模块的填充 mask，形状为 (#batch, 1, time)，
            (0, 0, 0) 表示伪 mask。
        :param att_cache: 自注意力的缓存张量，
            形状为 (#batch=1, head, cache_t1, d_k * 2)，head * d_k == size。
        :param cnn_cache: Conformer 层的卷积缓存，
            形状为 (#batch=1, size, cache_t2)。

        :return:
            - 输出张量 (#batch, time, size)
            - mask 张量 (#batch, time, time)
            - 新的注意力缓存张量 (#batch=1, head, cache_t1 + time, d_k * 2)
            - 新的卷积缓存张量 (#batch=1, size, cache_t2)
        """
        
        # 是否使用macaron风格的前馈网络
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:  # 如果在子模块前使用归一化
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))  # 添加额外前馈模块的输出
            if not self.normalize_before:  # 如果在子模块后使用归一化
                x = self.norm_ff_macaron(x)

        # 多头自注意力模块
        residual = x
        if self.normalize_before:  # 如果在子模块前使用归一化
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)  # 加上自注意力模块的输出
        if not self.normalize_before:  # 如果在子模块后使用归一化
            x = self.norm_mha(x)

        # 卷积模块
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:  # 如果在子模块前使用归一化
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)  # 计算卷积模块的输出
            x = residual + self.dropout(x)  # 加上卷积模块的输出

            if not self.normalize_before:  # 如果在子模块后使用归一化
                x = self.norm_conv(x)

        # 前馈模块
        residual = x
        if self.normalize_before:  # 如果在子模块前使用归一化
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))  # 加上前馈模块的输出
        if not self.normalize_before:  # 如果在子模块后使用归一化
            x = self.norm_ff(x)

        # 如果有卷积模块，则进行最终归一化
        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache  # 返回处理后的张量和相关缓存
