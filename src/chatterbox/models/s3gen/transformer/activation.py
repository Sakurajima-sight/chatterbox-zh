# Copyright (c) 2020 Johns Hopkins University (Shinji Watanabe)
#               2020 Northwestern Polytechnical University (Pengcheng Guo)
#               2020 Mobvoi Inc (Binbin Zhang)
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

"""Swish() 激活函数和 Snake 激活函数的实现，用于 Conformer 模型"""

import torch
from torch import nn, sin, pow
from torch.nn import Parameter


class Swish(torch.nn.Module):
    """构建 Swish 激活函数的对象"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现 Swish 激活函数。
        Swish 激活函数的定义为：f(x) = x * sigmoid(x)
        
        参数：
        - x: 输入张量，形状为 (B, C, T)
        
        返回：
        - 输出张量，形状与输入相同
        """
        return x * torch.sigmoid(x)


# 以下实现自 https://github.com/EdwardDixon/snake，使用 MIT 许可。
#   许可信息在 incl_licenses 目录中。
class Snake(nn.Module):
    """
    实现一个基于正弦的周期性激活函数（Snake）。
    该激活函数通过：x + 1/a * sin^2(x * a) 来计算输出。
    
    输入和输出形状：
        - 输入：形状为 (B, C, T)
        - 输出：与输入形状相同，形状为 (B, C, T)
    
    参数：
        - alpha: 一个可训练的参数（默认为 1.0）。该参数控制正弦的频率。
        - alpha_trainable: 是否训练 alpha（默认为 True）
        - alpha_logscale: 是否将 alpha 以对数尺度初始化（默认为 False）
    
    参考文献：
        - 该激活函数来自论文：Liu Ziyin, Tilman Hartwig, Masahito Ueda. "https://arxiv.org/abs/2006.08195"
    
    示例：
        >>> a1 = Snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        初始化 Snake 激活函数。
        
        参数：
            - in_features: 输入的特征数量（即输入张量的通道数）
            - alpha: 控制正弦函数频率的可训练参数，默认为 1.0
            - alpha_trainable: 是否使 alpha 可训练，默认为 True
            - alpha_logscale: 是否使用对数尺度来初始化 alpha，默认为 False
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # 初始化 alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # 如果是对数尺度，则初始化为零
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # 否则，初始化为一
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001  # 避免除以零

    def forward(self, x):
        """
        计算 Snake 激活函数的前向传递。

        对输入进行逐元素操作，计算：x + 1/a * sin^2(x * a)
        
        参数：
            - x: 输入张量，形状为 (B, C, T)
        
        返回：
            - 输出张量，形状与输入相同
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # 将 alpha 调整为 [B, C, T] 的形状
        if self.alpha_logscale:
            alpha = torch.exp(alpha)  # 如果 alpha 是对数尺度，则进行指数变换
        
        # 计算 Snake 激活函数公式：x + 1/a * sin^2(x * a)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
