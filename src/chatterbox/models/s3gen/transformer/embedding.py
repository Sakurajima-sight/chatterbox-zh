# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(torch.nn.Module):
    """
    位置编码模块，用于在序列模型（如Transformer）中加入位置信息。

    :param int d_model: 嵌入维度
    :param float dropout_rate: dropout 的概率
    :param int max_len: 最大输入长度
    :param bool reverse: 是否反转位置编码（默认不反转）

    位置编码计算方式：
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel))
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """
        构造位置编码对象

        :param d_model: 嵌入维度
        :param dropout_rate: dropout 的概率
        :param max_len: 最大输入长度
        :param reverse: 是否反转位置编码（默认不反转）
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)  # 缩放因子
        self.dropout = torch.nn.Dropout(p=dropout_rate)  # dropout层
        self.max_len = max_len

        # 初始化位置编码矩阵
        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))  # 除数项

        # 填充位置编码矩阵
        self.pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用 sin
        self.pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用 cos
        self.pe = self.pe.unsqueeze(0)  # 添加一个批量维度

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：为输入添加位置编码

        :param x: 输入张量，形状为 (batch, time, ...)
        :param offset: 位置偏移，支持int和tensor类型

        :return: 
            - 位置编码后的输入张量，形状为 (batch, time, ...)
            - 位置编码矩阵（用于兼容RelPositionalEncoding）
        """
        self.pe = self.pe.to(x.device)  # 将位置编码转移到输入张量的设备上
        pos_emb = self.position_encoding(offset, x.size(1), False)  # 获取位置编码
        x = x * self.xscale + pos_emb  # 将输入乘以缩放因子并加上位置编码
        return self.dropout(x), self.dropout(pos_emb)  # 返回加了位置编码的结果和位置编码本身

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """
        获取对应位置的编码，可以支持流式处理

        说明：
        在非流式场景下，我们只应用一次 dropout；但是在流式解码过程中，由于输入大小不断增加，我们将多次调用此函数，所以 dropout 会多次应用。

        :param offset: 起始位置偏移，可以是int或torch.tensor
        :param size: 需要的编码大小
        :param apply_dropout: 是否应用dropout，默认应用
        :return: 对应的编码矩阵
        """
        if isinstance(offset, int):
            assert offset + size <= self.max_len  # 检查偏移是否超出最大长度
            pos_emb = self.pe[:, offset:offset + size]  # 获取位置编码
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # 如果是标量
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:  # 针对批量流式解码
            assert torch.max(offset) + size <= self.max_len  # 确保偏移量合法
            index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)  # 计算索引
            flag = index > 0  # 生成一个标志，排除负偏移
            index = index * flag  # 移除负偏移
            pos_emb = F.embedding(index, self.pe[0])  # 执行位置编码查找

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)  # 应用dropout
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """
    相对位置编码模块，参考文献：[Appendix B in https://arxiv.org/abs/1901.02860]
    
    该模块用于计算相对位置编码，它不同于标准的绝对位置编码，在某些任务中，能够提高模型的性能。
    
    :param d_model: 嵌入维度
    :param dropout_rate: dropout 概率
    :param max_len: 最大输入长度
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """初始化类"""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)  # 继承PositionalEncoding类，设置反转为True

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算相对位置编码

        :param x: 输入张量，形状为 (batch, time, `*`)
        :param offset: 位置偏移量，可以是整数或张量
        
        :return: 
            - 位置编码后的张量 (batch, time, `*`)
            - 位置嵌入张量 (1, time, `*`)
        """
        self.pe = self.pe.to(x.device)  # 将位置编码矩阵移到输入张量所在设备
        x = x * self.xscale  # 对输入张量进行缩放
        pos_emb = self.position_encoding(offset, x.size(1), False)  # 获取位置编码
        return self.dropout(x), self.dropout(pos_emb)  # 返回dropout后的编码结果和位置编码


class WhisperPositionalEncoding(PositionalEncoding):
    """
    OpenAI Whisper模型中的正弦位置编码，用于处理语音或音频数据的输入。
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        """
        初始化WhisperPositionalEncoding模块

        :param d_model: 嵌入维度
        :param dropout_rate: dropout 概率
        :param max_len: 最大输入长度
        """
        super().__init__(d_model, dropout_rate, max_len)  # 调用父类构造函数
        self.xscale = 1.0  # Whisper模型不使用位置缩放因子
        # 计算对数时间尺度增量，按照论文中的公式生成正弦与余弦位置编码
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment *
                                   torch.arange(d_model // 2))  # 计算每个频率的逆时间尺度
        scaled_time = torch.arange(max_len)[:, np.newaxis] * \
            inv_timescales[np.newaxis, :]  # 生成与时间相关的缩放项
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # 生成正弦和余弦位置编码
        delattr(self, "pe")  # 删除原位置编码矩阵
        self.register_buffer("pe", pe.unsqueeze(0))  # 注册为缓冲区变量，避免在反向传播中更新


class LearnablePositionalEncoding(PositionalEncoding):
    """
    OpenAI Whisper模型中的可学习位置编码，通常用于编码器中。
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        """
        初始化LearnablePositionalEncoding模块

        :param d_model: 嵌入维度
        :param dropout_rate: dropout 概率
        :param max_len: 最大输入长度
        """
        super().__init__(d_model, dropout_rate, max_len)  # 调用父类构造函数
        # 覆盖父类的 pe，并将其设为可学习参数
        self.pe = torch.nn.Parameter(torch.empty(1, max_len, d_model))  # 使用可学习的参数来表示位置编码
        self.xscale = 1.0  # 不使用位置编码的缩放因子


class NoPositionalEncoding(torch.nn.Module):
    """
    不使用位置编码模块，仅返回零向量。
    该模块用于某些不需要位置编码的场景。
    """

    def __init__(self, d_model: int, dropout_rate: float):
        """
        初始化NoPositionalEncoding模块

        :param d_model: 嵌入维度
        :param dropout_rate: dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)  # dropout层

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回零向量以保持接口兼容性。

        :param x: 输入张量
        :param offset: 位置偏移量
        :return: 
            - dropout后的输入张量
            - 零位置编码张量
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)  # 创建零位置编码
        return self.dropout(x), pos_emb  # 返回dropout后的输入和零位置编码

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        """
        返回零位置编码（没有实际的位置编码）

        :param offset: 位置偏移
        :param size: 需要的位置编码大小
        :return: 返回一个零张量
        """
        return torch.zeros(1, size, self.d_model)  # 返回零的位置编码


class EspnetRelPositionalEncoding(torch.nn.Module):
    """
    相对位置编码模块（新实现）。

    该模块实现了相对位置编码，参见文献：Appendix B in https://arxiv.org/abs/1901.02860。
    详细信息可以参考https://github.com/espnet/espnet/pull/2816中的实现。

    :param d_model: 嵌入维度
    :param dropout_rate: dropout 概率
    :param max_len: 最大输入长度
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """
        初始化相对位置编码模块。

        :param d_model: 嵌入维度
        :param dropout_rate: dropout 的概率
        :param max_len: 最大输入长度
        """
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)  # 缩放因子
        self.dropout = torch.nn.Dropout(p=dropout_rate)  # dropout层
        self.pe = None  # 初始化位置编码为空
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))  # 初始化扩展位置编码

    def extend_pe(self, x: torch.Tensor):
        """
        扩展位置编码。

        用于重置位置编码矩阵。这里计算了正负位置编码的部分，并将其存储在self.pe中。

        :param x: 输入张量，用于扩展位置编码
        """
        if self.pe is not None:
            # self.pe 包含正负位置部分
            # self.pe 的长度为 2 * 输入长度 - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        # 创建正负位置编码
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        # 填充正负位置编码的sin和cos值
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # 翻转正位置编码并拼接正负位置编码
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        添加位置编码。

        :param x: 输入张量 (batch, time, `*`)

        :return: 
            - 编码后的张量 (batch, time, `*`)
            - 位置编码张量
        """
        self.extend_pe(x)  # 扩展位置编码
        x = x * self.xscale  # 缩放输入
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)  # 获取位置编码
        return self.dropout(x), self.dropout(pos_emb)  # 返回加了位置编码后的输入和位置编码

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        """
        获取流式位置编码。

        在流式处理时，位置编码需要动态计算，dropout 只会在整个序列级别上应用一次，但在流式解码过程中会多次调用此函数，因此dropout会被多次应用。

        :param offset: 起始偏移量，可以是int或torch.tensor
        :param size: 需要的位置编码的大小

        :return: 对应的编码
        """
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1: self.pe.size(1) // 2 + size,
        ]
        return pos_emb  # 返回根据偏移量和大小计算出来的相对位置编码
