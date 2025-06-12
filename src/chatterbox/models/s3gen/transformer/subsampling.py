# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Subsampling layer definition."""

from typing import Tuple, Union

import torch


class BaseSubsampling(torch.nn.Module):
    """
    基础下采样层基类，定义了基础的下采样操作。

    :param right_context: 右侧上下文
    :param subsampling_rate: 下采样率
    """

    def __init__(self):
        super().__init__()
        self.right_context = 0  # 右侧上下文
        self.subsampling_rate = 1  # 下采样率

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        """
        获取位置编码。

        :param offset: 偏移量
        :param size: 需要的位置编码的大小
        :return: 对应的编码
        """
        return self.pos_enc.position_encoding(offset, size)


class EmbedinigNoSubsampling(BaseSubsampling):
    """
    不进行下采样的输入嵌入层。
    该层将输入通过嵌入层映射到输出维度，并加入位置编码。
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """
        构造不进行下采样的嵌入层对象。

        :param idim: 输入维度
        :param odim: 输出维度
        :param dropout_rate: dropout 概率
        :param pos_enc_class: 位置编码类
        """
        super().__init__()
        self.embed = torch.nn.Embedding(idim, odim)  # 嵌入层
        self.pos_enc = pos_enc_class  # 位置编码

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播函数。

        :param x: 输入张量，形状为 (#batch, time, idim)
        :param x_mask: 输入的mask，形状为 (#batch, 1, time)
        :param offset: 位置偏移量，默认是0

        :return: 
            - 线性变换后的输入张量（#batch, time', odim），
              其中 time' = time（不做下采样）
            - 位置编码后的张量
            - 输入mask
        """
        x = self.embed(x)  # 通过嵌入层进行映射
        x, pos_emb = self.pos_enc(x, offset)  # 加入位置编码
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """
    不进行下采样的线性变换层。
    该层将输入通过线性层转换到输出维度，并加入位置编码。
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """
        构造不进行下采样的线性变换层对象。

        :param idim: 输入维度
        :param odim: 输出维度
        :param dropout_rate: dropout 概率
        :param pos_enc_class: 位置编码类
        """
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),  # 线性变换
            torch.nn.LayerNorm(odim, eps=1e-5),  # 层归一化
            torch.nn.Dropout(dropout_rate),  # dropout层
        )
        self.pos_enc = pos_enc_class  # 位置编码
        self.right_context = 0  # 右侧上下文
        self.subsampling_rate = 1  # 下采样率

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播函数。

        :param x: 输入张量，形状为 (#batch, time, idim)
        :param x_mask: 输入的mask，形状为 (#batch, 1, time)
        :param offset: 位置偏移量，默认是0

        :return:
            - 线性变换后的输入张量（#batch, time', odim），
              其中 time' = time（不做下采样）
            - 位置编码后的张量
            - 输入mask
        """
        x = self.out(x)  # 通过线性层、层归一化和dropout进行变换
        x, pos_emb = self.pos_enc(x, offset)  # 加入位置编码
        return x, pos_emb, x_mask


class Conv1dSubsampling2(BaseSubsampling):
    """
    1D卷积下采样层（将时间长度下采样为原来的一半），用于 Whisper 模型。
    该层通过一系列卷积操作将输入序列的时间维度缩减为原来的 1/2。

    :param idim: 输入维度。
    :param odim: 输出维度。
    :param dropout_rate: dropout 的概率。
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """
        构造一个 Conv1dSubsampling2 对象。

        :param idim: 输入维度。
        :param odim: 输出维度。
        :param dropout_rate: dropout 概率。
        :param pos_enc_class: 位置编码类，用于序列中的位置编码。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, kernel_size=3, padding=1),  # 第一层卷积
            torch.nn.GELU(),  # 激活函数
            torch.nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),  # 第二层卷积，stride=2实现下采样
            torch.nn.GELU(),  # 激活函数
        )
        self.pos_enc = pos_enc_class  # 位置编码
        # 每个卷积层的右侧上下文计算方式为： (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2  # 下采样率为2
        # 右侧上下文大小
        self.right_context = 4  # 4 = (3 - 1) * 1 + (3 - 1) * 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对输入 x 进行下采样。

        :param x: 输入张量，形状为 (#batch, time, idim)
        :param x_mask: 输入的 mask，形状为 (#batch, 1, time)
        :param offset: 位置偏移量，默认为 0

        :return: 
            - 下采样后的张量（#batch, time', odim），
              其中 time' = time // 2。
            - 下采样后的 mask（#batch, 1, time'），
              其中 time' = time // 2。
            - 位置编码。
        """
        time = x.size(1)
        x = x.transpose(1, 2)  # (b, f, t)，将 time 和 feature 维度交换
        x = self.conv(x)  # 通过卷积层进行处理
        x = x.transpose(1, 2)  # (b, t, f)，将 feature 和 time 维度交换回来
        x, pos_emb = self.pos_enc(x, offset)  # 加入位置编码
        return x, pos_emb, x_mask[:, :, (time + 1) % 2::2]  # 处理后的 mask，time' = time // 2


class Conv2dSubsampling4(BaseSubsampling):
    """
    2D卷积下采样层（将时间长度下采样为原来的一半，再下采样一次，最终时间长度为原来的 1/4）。
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """
        构造一个 Conv2dSubsampling4 对象。

        :param idim: 输入维度。
        :param odim: 输出维度。
        :param dropout_rate: dropout 概率。
        :param pos_enc_class: 位置编码类，用于序列中的位置编码。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),  # 第一个卷积操作，stride=2进行下采样
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Conv2d(odim, odim, 3, 2),  # 第二个卷积操作，stride=2进一步进行下采样
            torch.nn.ReLU(),  # 激活函数
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)  # 对卷积后的输出进行线性变换
        )
        self.pos_enc = pos_enc_class  # 位置编码
        self.subsampling_rate = 4  # 下采样率为4
        # 右侧上下文大小计算方式：6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对输入 x 进行下采样。

        :param x: 输入张量，形状为 (#batch, time, idim)
        :param x_mask: 输入的 mask，形状为 (#batch, 1, time)
        :param offset: 位置偏移量，默认为 0

        :return: 
            - 下采样后的张量（#batch, time', odim），
              其中 time' = time // 4。
            - 下采样后的 mask（#batch, 1, time'），
              其中 time' = time // 4。
            - 位置编码。
        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)，在输入张量的第二维添加通道维度
        x = self.conv(x)  # 通过卷积层进行处理
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))  # 线性变换并恢复为 (b, t, c * f) 形状
        x, pos_emb = self.pos_enc(x, offset)  # 加入位置编码
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]  # 处理后的 mask，time' = time // 4


class Conv2dSubsampling6(BaseSubsampling):
    """
    2D卷积下采样层（将时间长度下采样为原来的 1/6）。
    
    该层通过连续的卷积操作将输入序列的时间维度下采样到原来的 1/6。
    
    :param idim: 输入维度。
    :param odim: 输出维度。
    :param dropout_rate: dropout 概率。
    :param pos_enc_class: 位置编码类，用于序列中的位置编码。
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """
        构造一个 Conv2dSubsampling6 对象。

        :param idim: 输入维度。
        :param odim: 输出维度。
        :param dropout_rate: dropout 概率。
        :param pos_enc_class: 位置编码类。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),  # 第一个卷积层，stride=2进行下采样
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Conv2d(odim, odim, 5, 3),  # 第二个卷积层，stride=3进一步下采样
            torch.nn.ReLU(),  # 激活函数
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)  # 对卷积后的输出进行线性变换
        self.pos_enc = pos_enc_class  # 位置编码
        self.subsampling_rate = 6  # 下采样率为6
        # 右侧上下文大小计算： 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.right_context = 10  # 右侧上下文为10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对输入 x 进行下采样。

        :param x: 输入张量，形状为 (#batch, time, idim)
        :param x_mask: 输入的 mask，形状为 (#batch, 1, time)
        :param offset: 位置偏移量，默认为 0

        :return: 
            - 下采样后的张量（#batch, time', odim），
              其中 time' = time // 6。
            - 下采样后的 mask（#batch, 1, time'），
              其中 time' = time // 6。
            - 位置编码。
        """
        x = x.unsqueeze(1)  # (b, c, t, f)，在输入张量的第二维添加通道维度
        x = self.conv(x)  # 通过卷积层进行处理
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))  # 线性变换并恢复为 (b, t, c * f) 形状
        x, pos_emb = self.pos_enc(x, offset)  # 加入位置编码
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]  # 处理后的 mask，time' = time // 6


class Conv2dSubsampling8(BaseSubsampling):
    """
    2D卷积下采样层（将时间长度下采样为原来的1/8）。

    该层通过连续的卷积操作将输入序列的时间维度下采样到原来的 1/8。
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """
        构造一个 Conv2dSubsampling8 对象。

        :param idim: 输入维度。
        :param odim: 输出维度。
        :param dropout_rate: dropout 概率。
        :param pos_enc_class: 位置编码类。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),  # 第一个卷积层，stride=2进行下采样
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Conv2d(odim, odim, 3, 2),  # 第二个卷积层，stride=2进一步下采样
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Conv2d(odim, odim, 3, 2),  # 第三个卷积层，stride=2继续下采样
            torch.nn.ReLU(),  # 激活函数
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)  # 对卷积后的输出进行线性变换
        self.pos_enc = pos_enc_class  # 位置编码
        self.subsampling_rate = 8  # 下采样率为8
        # 右侧上下文大小计算：14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14  # 右侧上下文为14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对输入 x 进行下采样。

        :param x: 输入张量，形状为 (#batch, time, idim)
        :param x_mask: 输入的 mask，形状为 (#batch, 1, time)
        :param offset: 位置偏移量，默认为 0

        :return:
            - 下采样后的张量（#batch, time', odim），
              其中 time' = time // 8。
            - 下采样后的 mask（#batch, 1, time'），
              其中 time' = time // 8。
            - 位置编码。
        """
        x = x.unsqueeze(1)  # (b, c, t, f)，在输入张量的第二维添加通道维度
        x = self.conv(x)  # 通过卷积层进行处理
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))  # 线性变换并恢复为 (b, t, c * f) 形状
        x, pos_emb = self.pos_enc(x, offset)  # 加入位置编码
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]  # 处理后的 mask，time' = time // 8


class LegacyLinearNoSubsampling(BaseSubsampling):
    """
    不进行下采样的线性变换层。
    该层将输入通过线性层转换到输出维度，并加入位置编码。

    :param idim: 输入维度。
    :param odim: 输出维度。
    :param dropout_rate: dropout 的概率。
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """
        构造一个 LegacyLinearNoSubsampling 对象。

        :param idim: 输入维度。
        :param odim: 输出维度。
        :param dropout_rate: dropout 概率。
        :param pos_enc_class: 位置编码类，用于序列中的位置编码。
        """
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),  # 线性变换
            torch.nn.LayerNorm(odim, eps=1e-5),  # 层归一化
            torch.nn.Dropout(dropout_rate),  # dropout层
            torch.nn.ReLU(),  # 激活函数
        )
        self.pos_enc = pos_enc_class  # 位置编码
        self.right_context = 0  # 右侧上下文
        self.subsampling_rate = 1  # 不进行下采样，subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播函数。

        :param x: 输入张量，形状为 (#batch, time, idim)
        :param x_mask: 输入的mask，形状为 (#batch, 1, time)
        :param offset: 位置偏移量，默认为 0

        :return:
            - 下采样后的线性变换张量（#batch, time', odim），
              其中 time' = time（因为不进行下采样）
            - 位置编码后的张量
            - 输入的mask
        """
        x = self.out(x)  # 通过线性变换、层归一化、dropout和激活函数处理输入
        x, pos_emb = self.pos_enc(x, offset)  # 加入位置编码
        return x, pos_emb, x_mask  # 返回处理后的张量、位置编码和mask
