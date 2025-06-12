# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from .utils.mask import add_optional_chunk_mask
from .matcha.decoder import SinusoidalPosEmb, Block1D, ResnetBlock1D, Downsample1D, \
    TimestepEmbedding, Upsample1D
from .matcha.transformer import BasicTransformerBlock


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    将mask转换为偏置（bias）。用于在注意力计算中屏蔽部分输入。
    
    参数：
        mask: 一个布尔型的mask张量，用于标识哪些位置需要屏蔽。
        dtype: 输出张量的数据类型，支持torch.float32, torch.bfloat16, torch.float16。
    
    返回：
        偏置张量，用于attention计算中，将被屏蔽的位置置为一个很小的负值。
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    mask = (1.0 - mask) * -1.0e+10
    return mask



class Transpose(torch.nn.Module):
    """
    用于转置输入张量的维度，交换给定的两个维度。
    
    参数：
        dim0: 需要交换的第一个维度
        dim1: 需要交换的第二个维度
    """
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        """
        执行张量的维度交换操作。
        
        参数：
            x: 输入张量
        
        返回：
            转置后的张量
        """
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalBlock1D(Block1D):
    """
    具有因果卷积操作的Block1D类。因果卷积确保在计算时不会泄漏未来的时间步信息。
    
    参数：
        dim: 输入张量的维度
        dim_out: 输出张量的维度
    """
    def __init__(self, dim: int, dim_out: int):
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),  # 使用因果卷积
            Transpose(1, 2),                # 转置维度
            nn.LayerNorm(dim_out),         # 层归一化
            Transpose(1, 2),                # 恢复维度
            nn.Mish(),                      # 激活函数Mish
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        执行因果卷积并应用mask。mask用于指定需要处理的有效区域。
        
        参数：
            x: 输入张量
            mask: 屏蔽矩阵，用于在计算中屏蔽无效位置
        
        返回：
            经过处理后的输出张量
        """
        output = self.block(x * mask)  # 乘以mask来屏蔽不需要的部分
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    """
    具有因果卷积的残差块（ResnetBlock1D）。通过因果卷积构建的网络块，确保没有未来信息泄漏。
    
    参数：
        dim: 输入张量的维度
        dim_out: 输出张量的维度
        time_emb_dim: 时间嵌入的维度
        groups: 卷积的组数（默认为8）
    """
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)  # 第一个因果卷积块
        self.block2 = CausalBlock1D(dim_out, dim_out)  # 第二个因果卷积块


class CausalConv1d(torch.nn.Conv1d):
    """
    实现因果卷积操作的Conv1d类。因果卷积确保卷积核只考虑当前及以前的输入，避免信息泄漏。
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核的大小
        stride: 步幅（默认为1）
        dilation: 膨胀率（默认为1）
        groups: 卷积的组数（默认为1）
        bias: 是否使用偏置（默认为True）
        padding_mode: 填充方式（默认为'zeros'）
        device: 设备
        dtype: 数据类型
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None
                 ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride == 1  # 步幅必须为1
        self.causal_padding = (kernel_size - 1, 0)  # 因果卷积的填充策略

    def forward(self, x: torch.Tensor):
        """
        执行因果卷积操作。
        
        参数：
            x: 输入张量
        
        返回：
            经过因果卷积处理后的输出张量
        """
        x = F.pad(x, self.causal_padding)  # 对输入进行因果填充
        x = super(CausalConv1d, self).forward(x)  # 使用父类的卷积操作
        return x


class ConditionalDecoder(nn.Module):
    """
    该解码器模块实现了条件生成的UNet模型。该模型用于生成条件音频或图像数据，并支持时间步（timestep）嵌入。输入张量和目标张量的形状需要相同。
    如果文本内容比输出长度短或长，请在喂给解码器之前进行重新采样。
    """

    def __init__(
        self,
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
    ):
        """
        初始化解码器的超参数和网络结构。

        参数：
            in_channels: 输入的通道数（默认320）
            out_channels: 输出的通道数（默认80）
            causal: 是否使用因果卷积（默认True）
            channels: 中间层通道数列表（默认[256]）
            dropout: dropout比例（默认0.0）
            attention_head_dim: 注意力头的维度（默认64）
            n_blocks: 每个块的Transformer数量（默认4）
            num_mid_blocks: 中间块的数量（默认12）
            num_heads: 注意力头的数量（默认8）
            act_fn: 激活函数（默认"gelu"）
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.time_embeddings = SinusoidalPosEmb(in_channels)  # 时间嵌入层
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",  # 激活函数为silu
        )
        self.down_blocks = nn.ModuleList([])  # 下采样块
        self.mid_blocks = nn.ModuleList([])  # 中间块
        self.up_blocks = nn.ModuleList([])  # 上采样块

        # static_chunk_size目前未使用
        self.static_chunk_size = 0

        # 初始化down_blocks
        output_channel = in_channels
        for i in range(len(channels)):  
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim) if self.causal else \
                ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else
                CausalConv1d(output_channel, output_channel, 3) if self.causal else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # 初始化mid_blocks
        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim) if self.causal else \
                ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # 初始化up_blocks
        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = CausalResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            ) if self.causal else ResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3) if self.causal else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        
        # 最终的Block和投影层
        self.final_block = CausalBlock1D(channels[-1], channels[-1]) if self.causal else Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化网络权重。
        使用kaiming normal初始化卷积和线性层的权重，使用常数初始化偏置。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """
        该解码器的前向传播函数。
        
        参数：
            x (torch.Tensor): 输入张量，形状为(batch_size, in_channels, time)
            mask (torch.Tensor): 屏蔽张量，形状为(batch_size, 1, time)
            mu (torch.Tensor): 用于生成的条件张量
            t (torch.Tensor): 时间步张量，形状为(batch_size)
            spks (torch.Tensor, optional): 条件张量，形状为(batch_size, condition_channels)
            cond (torch.Tensor, optional): 预留用于未来的条件，默认为None
        
        返回：
            output: 经过解码的输出张量，形状为(batch_size, out_channels, time)
        """

        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)  # 时间嵌入

        x = pack([x, mu], "b * t")[0]

        # 如果spks不为None，则进行条件拼接
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        
        # 如果cond不为None，则进行条件拼接
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        
        # 下采样过程
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        
        masks = masks[:-1]
        mask_mid = masks[-1]

        # 中间块处理
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        # 上采样过程
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        
        # 最终处理和输出
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
