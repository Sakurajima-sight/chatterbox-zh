# Copyright (c) 2025 Resemble AI
# Author: Manmay Nakhashi
# MIT License
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class RelativePositionBias(nn.Module):
    """
    该类用于计算相对位置偏置，常用于自注意力机制中。
    相对位置偏置帮助模型理解序列中各元素之间的相对位置关系。
    """

    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        """
        初始化相对位置偏置的参数。

        :param scale: 缩放因子
        :param causal: 是否为因果（causal）偏置，通常用于解码器的自注意力
        :param num_buckets: 相对位置的桶数，用于对位置进行离散化
        :param max_distance: 最大位置距离，用于计算偏置
        :param heads: 注意力头的数量
        """
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)  # 初始化相对位置偏置的嵌入

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        """
        计算相对位置的桶值，将相对位置映射到一个离散的桶中。

        :param relative_position: 位置差异
        :param causal: 是否为因果
        :param num_buckets: 桶的数量
        :param max_distance: 最大距离
        :return: 映射后的桶值
        """
        ret = 0
        n = -relative_position  # 相对位置取反
        if not causal:
            num_buckets //= 2  # 非因果的情况下，桶数减半
            ret += (n < 0).long() * num_buckets  # 如果n小于0，设置为负半桶
            n = torch.abs(n)  # 对n取绝对值
        else:
            n = torch.max(n, torch.zeros_like(n))  # 如果是因果偏置，则n小于0时取0

        max_exact = num_buckets // 2  # 最大精确桶的数量
        is_small = n < max_exact  # 判断n是否小于最大精确桶数

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()  # 对于较大的n，进行平滑处理
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))  # 限制最大值

        ret += torch.where(is_small, n, val_if_large)  # 根据条件选择小值或大值
        return ret

    def forward(self, qk_dots):
        """
        计算并返回相对位置偏置。

        :param qk_dots: 注意力的QK点积，表示查询和键之间的相似度
        :return: 带有相对位置偏置的QK点积
        """
        i, j, device = *qk_dots.shape[-2:], qk_dots.device  # 获取QK点积的形状
        q_pos = torch.arange(i, dtype=torch.long, device=device)  # 生成查询位置
        k_pos = torch.arange(j, dtype=torch.long, device=device)  # 生成键位置
        rel_pos = k_pos[None, :] - q_pos[:, None]  # 计算位置差异
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)  # 将位置差异映射到桶
        values = self.relative_attention_bias(rp_bucket)  # 获取相对位置偏置的嵌入
        bias = rearrange(values, 'i j h -> () h i j')  # 重排偏置值
        return qk_dots + (bias * self.scale)  # 加上偏置并进行缩放


class AttentionQKV(nn.Module):
    """
    该类实现了多头注意力机制中的QKV计算，支持标准的缩放点积注意力以及Flash Attention。
    """

    def __init__(self, n_heads, head_dim, dropout_rate=0.1, scale=None, flash=False):
        """
        初始化多头注意力模块。

        :param n_heads: 注意力头的数量
        :param head_dim: 每个注意力头的维度
        :param dropout_rate: Dropout率
        :param scale: 缩放因子
        :param flash: 是否使用Flash Attention（高效注意力计算）
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5  # 缩放因子默认为head_dim的倒数
        self.flash = flash  # 是否使用Flash Attention
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层
        self.flash_config = self.setup_flash_config() if flash else None  # Flash Attention配置

    def setup_flash_config(self):
        """
        设置Flash Attention的配置。
        """
        flash_config = {
            'enable_flash': True,
            'enable_math': True,
            'enable_mem_efficient': True
        }
        return flash_config

    def forward(self, q, k, v, mask=None):
        """
        前向传播，计算QKV并返回注意力结果。

        :param q: 查询（Query）
        :param k: 键（Key）
        :param v: 值（Value）
        :param mask: 注意力掩码
        :return: 注意力结果
        """
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]  # 将QKV分头
        if self.flash:
            out = self.flash_attention(q, k, v, mask=mask)  # 如果使用Flash Attention，则计算Flash Attention
        else:
            out = self.scaled_dot_product_attention(q, k, v, mask=mask)  # 否则使用标准的缩放点积注意力

        return self.combine_heads(out)  # 合并头部并返回

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        计算标准的缩放点积注意力。

        :param q: 查询（Query）
        :param k: 键（Key）
        :param v: 值（Value）
        :param mask: 掩码
        :return: 注意力输出
        """
        sim = torch.einsum("bhlt,bhls->bhts", q, k) * self.scale  # 计算Q和K的点积并缩放
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float('-inf'))  # 如果存在mask，填充为负无穷
        attn = torch.softmax(sim, dim=-1)  # 计算注意力权重
        attn = self.dropout(attn)  # 应用Dropout
        return torch.einsum("bhts,bhls->bhlt", attn, v)  # 计算注意力输出

    def flash_attention(self, q, k, v, mask=None):
        """
        计算Flash Attention。

        :param q: 查询（Query）
        :param k: 键（Key）
        :param v: 值（Value）
        :param mask: 掩码
        :return: 注意力输出
        """
        config = self.flash_config if self.flash_config else {}
        with torch.backends.cuda.sdp_kernel(**config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.
            )
        return out

    def split_heads(self, x):
        """
        将输入张量分割成多个头。

        :param x: 输入张量
        :return: 分割后的张量
        """
        bs, length, _ = x.shape
        x = x.view(bs, length, self.n_heads, self.head_dim)  # 重塑张量
        return x.permute(0, 2, 1, 3)  # 转置以便分割头部

    def combine_heads(self, x):
        """
        合并多个注意力头的结果。

        :param x: 输入张量
        :return: 合并后的张量
        """
        bs, _, length, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # 转置并确保张量连续
        return x.view(bs, length, -1)  # 合并多个头并返回


class AttentionBlock2(nn.Module):
    """
    一个注意力模块，使得空间位置可以互相关注，使用AttentionQKV和针对Q、K、V的单独线性变换。
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        relative_pos_embeddings=False,
        flash_attention=True,
        dropout_rate=0.2,
        scale=None
    ):
        """
        初始化AttentionBlock2模块。

        :param channels: 输入的通道数
        :param num_heads: 注意力头的数量
        :param num_head_channels: 每个注意力头的通道数（默认为-1，表示自动计算）
        :param relative_pos_embeddings: 是否使用相对位置嵌入
        :param flash_attention: 是否使用Flash Attention（高效注意力计算）
        :param dropout_rate: Dropout的比例
        :param scale: 缩放因子
        """
        super().__init__()
        self.channels = channels

        # 如果指定了num_head_channels，则确保channels可以被num_head_channels整除
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)  # 对输入进行LayerNorm规范化

        # 为Q、K、V分别创建线性变换层
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        # 初始化注意力机制
        self.attention = AttentionQKV(self.num_heads, channels // self.num_heads, dropout_rate=dropout_rate, flash=flash_attention, scale=scale)

        # 输出投影层
        self.proj_out = nn.Linear(channels, channels)

        # 如果使用相对位置嵌入，初始化相对位置偏置
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x1, x2, mask=None):
        """
        前向传播：对两个输入张量应用注意力操作。

        :param x1: 输入张量1，通常作为查询Q
        :param x2: 输入张量2，通常作为键K和值V
        :param mask: 掩码（可选），用于忽略某些位置的计算
        :return: 加权和后的结果
        """
        b1, c1, *spatial1 = x1.shape  # 获取x1的批次大小和通道数
        b2, c2, *spatial2 = x2.shape  # 获取x2的批次大小和通道数

        x1_norm = self.norm(x1)  # 对x1进行规范化
        x2_norm = self.norm(x2)  # 对x2进行规范化

        # 通过线性变换得到Q、K、V
        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        # 使用注意力机制计算输出
        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)  # 对输出进行投影

        # 返回最终的结果，形状保持与x1一致
        return (x1 + h).reshape(b1, c1, *spatial1)


class Perceiver(nn.Module):
    """参照论文：https://arxiv.org/abs/2103.03206"""

    def __init__(self, pre_attention_query_token=32, pre_attention_query_size=1024, embedding_dim=1024, num_attn_heads=4):
        """
        初始化Perceiver模块。

        :param pre_attention_query_token: 预注意力的查询令牌数量
        :param pre_attention_query_size: 每个查询令牌的大小
        :param embedding_dim: 嵌入空间的维度
        :param num_attn_heads: 注意力头的数量
        """
        super().__init__()

        # 初始化预注意力查询参数
        self.pre_attention_query = torch.nn.Parameter(
            torch.empty(1, pre_attention_query_token, pre_attention_query_size)
        )

        # 计算查询的均匀初始化方差
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (pre_attention_query_token + pre_attention_query_token))

        # 使用均匀分布初始化预注意力查询
        self.pre_attention_query.data.uniform_(-query_variance, query_variance)

        # 初始化注意力块
        self.attn = AttentionBlock2(embedding_dim, num_attn_heads)

    def forward(self, h):
        """
        Perceiver模块的前向传播。

        :param h: 输入张量
        :return: 经过注意力机制后的输出
        """
        # 扩展预注意力查询以匹配输入的批次大小
        query_ = self.pre_attention_query.expand(h.shape[0], -1, -1)
        # 应用第一个注意力机制（交叉注意力）
        pre_att = self.attn(query_, h)
        # 应用第二个注意力机制（自注意力）
        attn = self.attn(pre_att, pre_att)
        return attn
