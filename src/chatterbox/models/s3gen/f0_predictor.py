# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
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

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class ConvRNNF0Predictor(nn.Module):
    """
    该类实现了一个基于卷积网络的预测器，用于生成F0（基频）值。
    它通过多个卷积层提取输入特征，并通过全连接层进行分类，最终预测出F0值。
    """

    def __init__(self,
                 num_class: int = 1,       # 预测类别数，默认为1，用于回归任务
                 in_channels: int = 80,    # 输入通道数，默认为80
                 cond_channels: int = 512  # 条件网络通道数，默认为512
                 ):
        """
        初始化ConvRNNF0Predictor网络的结构和参数。

        参数：
            num_class: 预测类别数（默认为1）
            in_channels: 输入特征的通道数（默认为80）
            cond_channels: 条件网络中间层的通道数（默认为512）
        """
        super().__init__()

        self.num_class = num_class  # 预测的类别数
        # 条件网络（condnet）包含多个卷积层，通过卷积提取特征
        self.condnet = nn.Sequential(
            weight_norm(  # 使用权重归一化来增强训练的稳定性
                nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),  # 使用ELU激活函数
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
        )
        
        # 全连接层，进行最终的F0分类（或回归）
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数：输入音频特征张量，经过条件网络处理，并通过分类器输出预测的F0值。

        参数：
            x (torch.Tensor): 输入张量，形状为(batch_size, in_channels, time)
        
        返回：
            torch.Tensor: 预测的F0值，形状为(batch_size, num_class)，取绝对值作为输出
        """
        x = self.condnet(x)  # 通过条件网络提取特征
        x = x.transpose(1, 2)  # 转置维度以匹配全连接层的输入格式
        return torch.abs(self.classifier(x).squeeze(-1))  # 通过全连接层预测F0，并去除多余的维度，取绝对值
