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
"""ConvolutionModule 定义"""

from typing import Tuple

import torch
from torch import nn


class ConvolutionModule(nn.Module):
    """Conformer 模型中的卷积模块。"""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """构造 ConvolutionModule 对象。

        参数：
            channels (int): 卷积层的通道数。
            kernel_size (int): 卷积核的大小，默认为 15。
            activation (nn.Module): 激活函数，默认为 ReLU。
            norm (str): 归一化类型，可以是 "batch_norm" 或 "layer_norm"。
            causal (bool): 是否使用因果卷积（默认为 False）。
            bias (bool): 是否在卷积中使用偏置，默认为 True。
        """
        super().__init__()

        # 第一层点卷积
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # 判断是否使用因果卷积
        if causal:
            padding = 0
            self.lorder = kernel_size - 1  # 因果卷积的左侧长度
        else:
            # 对于非因果卷积，卷积核大小应该是奇数
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2  # 对称卷积的填充
            self.lorder = 0

        # 深度卷积层
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,  # 深度卷积，每个通道独立卷积
            bias=bias,
        )

        # 根据归一化类型选择 BatchNorm 或 LayerNorm
        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        # 第二层点卷积
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation  # 激活函数

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算卷积模块的输出。

        参数：
            x (torch.Tensor): 输入张量，形状为 (#batch, time, channels)。
            mask_pad (torch.Tensor): 用于批量填充的掩码，形状为 (#batch, 1, time)，
                (0, 0, 0) 表示假的掩码。
            cache (torch.Tensor): 用于因果卷积的左侧上下文缓存，形状为 (#batch, channels, cache_t)，
                (0, 0, 0) 表示假的缓存。

        返回：
            torch.Tensor: 输出张量，形状为 (#batch, time, channels)。
            torch.Tensor: 新的缓存张量。
        """
        # 交换时间维度和特征维度
        x = x.transpose(1, 2)  # 变为 (#batch, channels, time)

        # 根据掩码填充批次
        if mask_pad.size(2) > 0:  # 如果时间维度大于0
            x.masked_fill_(~mask_pad, 0.0)

        # 如果是因果卷积，需要处理缓存
        if self.lorder > 0:
            if cache.size(2) == 0:  # 如果缓存为空
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # 确保批量大小一致
                assert cache.size(1) == x.size(1)  # 确保通道数一致
                x = torch.cat((cache, x), dim=2)  # 拼接缓存和当前输入
            new_cache = x[:, :, -self.lorder:]  # 获取新的缓存
        else:
            # 如果不需要缓存，返回一个假的张量
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # 使用 GLU 机制
        x = self.pointwise_conv1(x)  # (batch, 2*channel, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, time)

        # 进行 1D 深度卷积
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))  # 激活函数与归一化
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)  # 第二次点卷积

        # 根据掩码填充批次
        if mask_pad.size(2) > 0:  # 如果时间维度大于0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache  # 返回转置后的输出和新缓存
