# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .convolution import ConvolutionModule
from .encoder_layer import ConformerEncoderLayer
from .positionwise_feed_forward import PositionwiseFeedForward
from ..utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from ..utils.mask import make_pad_mask
from ..utils.mask import add_optional_chunk_mask


class Upsample1D(nn.Module):
    """
    A 1D upsampling layer with an optional convolution.
    
    参数:
        channels (`int`):
            输入和输出的通道数。
        use_conv (`bool`, default `False`):
            是否使用卷积。
        use_conv_transpose (`bool`, default `False`):
            是否使用反卷积。
        out_channels (`int`, optional):
            输出通道数，默认为`channels`。
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # 在这种模式下，首先重复插值，然后进行步幅为1的卷积
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        """
        inputs: 输入的张量，形状为(batch_size, seq_len, channels)
        input_lengths: 输入序列的长度
        """
        # 先进行上采样
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        # 填充操作，保证输出的长度一致
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        # 使用卷积
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    """
    A layer that performs pre-lookahead convolution.

    参数:
        channels (`int`):
            输入和输出的通道数。
        pre_lookahead_len (`int`, default `1`):
            预先前瞻的长度。
    """

    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        该层的输入是(batch_size, seq_len, channels)的张量
        """
        # 转置输入数据的维度，使得卷积操作适用
        outputs = inputs.transpose(1, 2).contiguous()
        # 前瞻操作
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # 卷积操作
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # 残差连接
        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoder(torch.nn.Module):
    """
    这是一个使用上采样的 Conformer 编码器，用于处理序列数据。它结合了常规的 Conformer 编码器和上采样层，
    可以用于音频信号的处理，如语音识别等任务。

    参数：
        input_size (int): 输入维度
        output_size (int): 注意力输出的维度
        attention_heads (int): 多头注意力的头数
        linear_units (int): 位置前馈网络的隐藏单元数
        num_blocks (int): 编码器块的数量
        dropout_rate (float): dropout 比率
        attention_dropout_rate (float): 注意力层的 dropout 比率
        positional_dropout_rate (float): 添加位置编码后的 dropout 比率
        input_layer (str): 输入层类型，可选 [linear, conv2d, conv2d6, conv2d8]
        pos_enc_layer_type (str): 编码器位置编码层类型，可选 [abs_pos, scaled_abs_pos, rel_pos, no_pos]
        normalize_before (bool): 是否在每个子模块之前使用层归一化
        static_chunk_size (int): 用于静态块训练和解码的块大小
        use_dynamic_chunk (bool): 是否使用动态块大小进行训练
        global_cmvn (Optional[torch.nn.Module]): 可选的全局 CMVN 模块
        use_dynamic_left_chunk (bool): 是否使用动态左块
        positionwise_conv_kernel_size (int): 位置卷积核的大小
        macaron_style (bool): 是否使用 Macaron 风格
        selfattention_layer_type (str): 自注意力层类型
        activation_type (str): 激活函数类型
        use_cnn_module (bool): 是否使用卷积模块
        cnn_module_kernel (int): 卷积模块的核大小
        causal (bool): 是否使用因果卷积
        cnn_module_norm (str): 卷积模块的归一化方式
        key_bias (bool): 是否在注意力中使用偏置
        gradient_checkpointing (bool): 是否使用梯度检查点
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        # 嵌入层：输入到输出的特征转换和位置编码
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        # 激活函数
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()

        # 自注意力模块定义
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # 前馈网络模块定义
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # 卷积模块定义
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        # 前瞻层
        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)

        # 编码器层
        self.encoders = torch.nn.ModuleList([ 
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])

        # 上采样层
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)
        self.up_embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )

        # 上采样后的编码器层
        self.up_encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(4)
        ])

    def output_size(self) -> int:
        """返回编码器的输出维度"""
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数，处理输入张量，经过编码器的处理。

        参数:
            xs: 填充后的输入张量，形状为(B, T, D)
            xs_lens: 输入长度，形状为(B)
            decoding_chunk_size: 解码时的块大小（对于动态块训练有用）
            num_decoding_left_chunks: 解码时的剩余块数量

        返回:
            encoder 输出的张量 xs 和经过下采样后的掩码
            xs: 填充后的输出张量 (B, T' ~= T/subsample_rate, D)
            masks: 填充掩码 (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        # 前瞻 + Conformer 编码器
        xs = self.pre_lookahead_layer(xs)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        # 上采样 + Conformer 编码器
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size * self.up_layer.stride,
                                              num_decoding_left_chunks)
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        """对输入进行多个编码器层的前向传播"""
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                          pos_emb: torch.Tensor,
                          mask_pad: torch.Tensor) -> torch.Tensor:
        """对上采样后的输入进行多个编码器层的前向传播"""
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs
