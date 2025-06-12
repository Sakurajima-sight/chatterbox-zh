import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat

from .transformer import BasicTransformerBlock


class SinusoidalPosEmb(torch.nn.Module):
    """
    该类用于实现正弦位置嵌入（Sinusoidal Positional Embedding），
    它通过正弦和余弦函数生成不同频率的嵌入向量，常用于Transformer架构中提供位置信息。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        # 对输入张量x进行正弦位置嵌入处理
        if x.ndim < 1:
            x = x.unsqueeze(0)  # 确保x的维度至少为1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    """
    一个1D卷积块，通过卷积、分组归一化和激活函数进行操作。
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        # 对输入张量x进行处理，应用mask进行遮掩
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    """
    一个包含残差连接的1D卷积块，支持时间嵌入的输入。
    """

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        # 计算残差块的输出
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    """
    1D下采样层，通过卷积将输入张量的时间步长减少一半。
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        # 对输入进行下采样
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    """
    时间步嵌入模块，用于将时间信息转化为嵌入向量，通常用于条件生成任务。
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        # 计算时间步嵌入
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """
    1D上采样层，支持使用卷积转置或常规卷积进行上采样。
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        # 对输入进行上采样
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class ConformerWrapper(ConformerBlock):
    """
    该类是ConformerBlock的包装类，主要用于简化和扩展Conformer模块的功能。
    它继承了ConformerBlock，并且在初始化时对一些参数进行了定制。
    """

    def __init__(self,  # pylint: disable=useless-super-delegation
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        """
        初始化ConformerWrapper类，调用父类（ConformerBlock）的构造函数来进行初始化。

        参数:
        - dim: 特征维度
        - dim_head: 每个注意力头的维度
        - heads: 注意力头的数量
        - ff_mult: 前馈网络的扩展因子
        - conv_expansion_factor: 卷积扩展因子
        - conv_kernel_size: 卷积核大小
        - attn_dropout: 注意力dropout概率
        - ff_dropout: 前馈网络dropout概率
        - conv_dropout: 卷积层dropout概率
        - conv_causal: 是否启用因果卷积
        """
        # 调用父类的构造函数进行初始化
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        """
        前向传播函数。

        参数:
        - hidden_states: 输入的隐藏状态
        - attention_mask: 注意力掩码
        - encoder_hidden_states: 编码器的隐藏状态（可选）
        - encoder_attention_mask: 编码器的注意力掩码（可选）
        - timestep: 时间步（可选）

        返回:
        - 通过ConformerBlock的前向传播计算得到的输出
        """
        # 调用父类的前向传播方法，传递输入和掩码
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class Decoder(nn.Module):
    """
    该类实现了一个包含下采样、编码、和上采样的解码器模块，常用于生成模型中。解码器将输入的潜在表示逐步转换为输出，通过一系列的卷积块和Transformer块实现对输入的处理。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
    ):
        """
        初始化Decoder模块。

        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - channels: 每个阶段的通道数
        - dropout: dropout概率
        - attention_head_dim: 每个注意力头的维度
        - n_blocks: 每个模块的块数
        - num_mid_blocks: 中间模块的块数
        - num_heads: 注意力头的数量
        - act_fn: 激活函数
        - down_block_type: 下采样模块的类型
        - mid_block_type: 中间模块的类型
        - up_block_type: 上采样模块的类型
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # 下采样模块
        self.down_blocks = nn.ModuleList([])

        # 中间模块
        self.mid_blocks = nn.ModuleList([])

        # 上采样模块
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn):
        """
        根据指定的块类型返回相应的模块。

        参数:
        - block_type: 块类型（"conformer" 或 "transformer"）
        - dim: 输入和输出的通道数
        - attention_head_dim: 每个注意力头的维度
        - num_heads: 注意力头的数量
        - dropout: dropout的概率
        - act_fn: 激活函数的类型

        返回:
        - 返回一个块（Conformer或Transformer块）
        """
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        """
        初始化网络权重。
        对于卷积层、线性层以及分组归一化层，使用不同的初始化方法。
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
        解码器的前向传播函数，将输入经过多个模块（下采样、中间块、上采样）处理，最后生成输出。

        参数:
        - x (torch.Tensor): 输入的隐藏状态，形状为 (batch_size, in_channels, time)
        - mask (torch.Tensor): 掩码，形状为 (batch_size, 1, time)
        - mu (torch.Tensor): 用于调节生成的潜在表示
        - t (torch.Tensor): 时间步（通常用于生成任务中）
        - spks (torch.Tensor, optional): 可选的条件信息，形状为 (batch_size, condition_channels)
        - cond (torch.Tensor, optional): 可选的条件参数，未来可以扩展使用

        返回:
        - output (torch.Tensor): 经过处理的输出，形状为 (batch_size, out_channels, time)
        """

        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # 保存跳跃连接的隐藏状态
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask
