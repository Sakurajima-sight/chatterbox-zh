import math

import torch
import torch.nn as nn
from einops import rearrange


# 生成序列掩码，用于忽略输入中无效的位置
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

# 定义层归一化（Layer Normalization）模块
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        """
        LayerNorm类实现了层归一化功能，接收的输入是一个多维张量，按照最后一个维度进行归一化。
        :param channels: 输入的通道数
        :param eps: 防止除零的一个小常数
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        # 可学习的gamma和beta参数
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """
        前向传播方法，进行标准化操作。
        :param x: 输入张量
        :return: 归一化后的张量
        """
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)  # 计算均值
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)  # 计算方差

        # 归一化公式
        x = (x - mean) * torch.rsqrt(variance + self.eps)

        # 调整gamma和beta的形状以便与输入张量匹配
        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# 定义卷积-ReLU-归一化模块
class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        """
        ConvReluNorm类结合了卷积层、ReLU激活层和归一化层，形成一个卷积-激活-归一化的模块。
        :param in_channels: 输入通道数
        :param hidden_channels: 隐藏通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param n_layers: 卷积层的层数
        :param p_dropout: Dropout的概率
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        # 第一个卷积层
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        
        # 后续的卷积层
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        
        # 最后一个1x1卷积层
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        """
        前向传播方法，依次通过卷积、归一化、激活和Dropout层。
        :param x: 输入张量
        :param x_mask: 掩码张量，用于屏蔽无效部分
        :return: 输出张量
        """
        x_org = x  # 保存原始输入
        for i in range(self.n_layers):
            # 通过卷积层
            x = self.conv_layers[i](x * x_mask)
            # 归一化
            x = self.norm_layers[i](x)
            # ReLU激活和Dropout
            x = self.relu_drop(x)
        # 残差连接
        x = x_org + self.proj(x)
        return x * x_mask  # 返回带有掩码的输出


# 定义时长预测器（Duration Predictor）模块
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        """
        DurationPredictor类用于预测输入序列中每个元素的时长。
        :param in_channels: 输入通道数
        :param filter_channels: 过滤通道数
        :param kernel_size: 卷积核大小
        :param p_dropout: Dropout的概率
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        # Dropout层
        self.drop = torch.nn.Dropout(p_dropout)
        # 第一层卷积
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        # 第二层卷积
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        # 最后输出层
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        """
        前向传播方法，经过两层卷积、归一化和Dropout，最后输出时长预测。
        :param x: 输入张量
        :param x_mask: 掩码张量，用于屏蔽无效部分
        :return: 预测的时长
        """
        # 第一层卷积
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)  # ReLU激活
        x = self.norm_1(x)
        x = self.drop(x)  # Dropout

        # 第二层卷积
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)  # ReLU激活
        x = self.norm_2(x)
        x = self.drop(x)  # Dropout

        # 输出时长预测
        x = self.proj(x * x_mask)
        return x * x_mask  # 返回带有掩码的预测结果

class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE模块

    Rotary编码通过在二维平面中旋转特征对来进行位置编码。 
    也就是说，它将$d$个特征组织为$\frac{d}{2}$对，每对特征可以看作二维平面中的一个坐标。
    该编码会根据标记的位置旋转这些坐标。
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        初始化RotaryPositionalEmbeddings类。
        :param d: 特征数$d$
        :param base: 用于计算旋转角度$\Theta$的常数
        """
        super().__init__()

        self.base = base  # base用于计算旋转角度
        self.d = int(d)  # 特征的维度
        self.cos_cached = None  # 存储缓存的cos值
        self.sin_cached = None  # 存储缓存的sin值

    def _build_cache(self, x: torch.Tensor):
        """
        缓存$\cos$和$\sin$值，避免每次计算时都进行重复计算
        :param x: 输入张量
        """
        # 如果缓存已构建且序列长度未变，则直接返回
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # 获取序列长度
        seq_len = x.shape[0]

        # 计算旋转角度 $\Theta = \theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # 创建位置索引 `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # 计算位置索引与$\theta_i$的乘积
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # 拼接成最终的旋转角度列表
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # 缓存cos和sin值
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        """
        计算旋转编码中的负半部分，即将输入的后半部分特征反转。
        :param x: 输入张量
        :return: 反转后的张量
        """
        # 计算特征的一半维度
        d_2 = self.d // 2

        # 计算$[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        前向传播方法，将旋转位置编码应用于输入的张量。
        :param x: 输入张量，形状为 `[seq_len, batch_size, n_heads, d]`
        :return: 经过旋转位置编码处理后的张量
        """
        # 调整张量形状，便于后续处理
        x = rearrange(x, "b h t d -> t b h d")

        # 构建cos和sin的缓存
        self._build_cache(x)

        # 分离输入特征，选择是否仅对一部分特征应用旋转位置编码
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        # 计算旋转编码的负半部分
        neg_half_x = self._neg_half(x_rope)

        # 应用旋转编码公式
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        # 合并处理后的张量，并恢复原始形状
        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention类实现了多头自注意力机制（Multi-Head Attention），
    通过多个头的注意力计算来捕捉不同的特征关系。适用于变压器架构中。
    """

    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        heads_share=True,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=False,
    ):
        """
        初始化MultiHeadAttention类。
        :param channels: 输入特征的通道数
        :param out_channels: 输出特征的通道数
        :param n_heads: 多头注意力中的头数
        :param heads_share: 是否共享所有头的参数
        :param p_dropout: Dropout的概率
        :param proximal_bias: 是否使用近端偏置（仅适用于自注意力）
        :param proximal_init: 是否使用初始化的近端偏置
        """
        super().__init__()
        assert channels % n_heads == 0  # 确保通道数可以被头数整除

        self.channels = channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.n_heads = n_heads  # 注意力头数
        self.heads_share = heads_share  # 是否共享头的参数
        self.proximal_bias = proximal_bias  # 是否使用近端偏置
        self.p_dropout = p_dropout  # Dropout的概率
        self.attn = None  # 注意力权重

        self.k_channels = channels // n_heads  # 每个头的通道数
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)  # 查询（Q）卷积层
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)  # 键（K）卷积层
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)  # 值（V）卷积层

        # 使用旋转位置编码（RoPE）
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)  # 输出卷积层
        self.drop = torch.nn.Dropout(p_dropout)  # Dropout层

        # 初始化卷积层权重
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        """
        前向传播方法，计算查询、键和值的注意力输出。
        :param x: 输入张量（通常是查询）
        :param c: 输入张量（通常是键和值）
        :param attn_mask: 可选的注意力掩码，屏蔽掉不需要关注的位置
        :return: 输出张量
        """
        # 计算查询、键和值
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        # 计算注意力
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        # 输出卷积
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        """
        计算查询、键和值的注意力分数和加权和。
        :param query: 查询张量
        :param key: 键张量
        :param value: 值张量
        :param mask: 可选的注意力掩码，屏蔽掉不需要关注的位置
        :return: 输出张量和注意力权重
        """
        # 获取张量的尺寸
        b, d, t_s, t_t = (*key.size(), query.size(2))
        
        # 调整张量形状
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        # 应用旋转位置编码
        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        # 如果使用近端偏置，计算并加到分数上
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        
        # 如果有掩码，应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # 计算softmax并应用Dropout
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        
        # 计算加权的值
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    @staticmethod
    def _attention_bias_proximal(length):
        """
        计算近端偏置，通常用于自注意力任务，表示位置间的距离影响。
        :param length: 序列长度
        :return: 计算出的近端偏置
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    """
    FFN类实现了前馈神经网络（Feed-Forward Network），
    该网络通常包含两个卷积层，中间带有ReLU激活和Dropout，用于处理输入特征。
    """

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        """
        初始化FFN类。
        :param in_channels: 输入特征的通道数
        :param out_channels: 输出特征的通道数
        :param filter_channels: 中间层的通道数（过滤通道数）
        :param kernel_size: 卷积核大小
        :param p_dropout: Dropout的概率
        """
        super().__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.filter_channels = filter_channels  # 中间层通道数
        self.kernel_size = kernel_size  # 卷积核大小
        self.p_dropout = p_dropout  # Dropout的概率

        # 定义卷积层
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = torch.nn.Dropout(p_dropout)  # Dropout层

    def forward(self, x, x_mask):
        """
        前向传播方法，通过卷积层和激活函数处理输入。
        :param x: 输入张量
        :param x_mask: 掩码张量，用于忽略无效位置
        :return: 输出张量
        """
        x = self.conv_1(x * x_mask)  # 先经过第一个卷积层
        x = torch.relu(x)  # ReLU激活
        x = self.drop(x)  # 应用Dropout
        x = self.conv_2(x * x_mask)  # 经过第二个卷积层
        return x * x_mask  # 返回掩码后的输出


class Encoder(nn.Module):
    """
    Encoder类实现了一个编码器，通常用于Transformer模型中。
    编码器由多层注意力机制（Multi-Head Attention）和前馈神经网络（FFN）组成，并使用层归一化（LayerNorm）进行处理。
    """

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        **kwargs,
    ):
        """
        初始化Encoder类。
        :param hidden_channels: 隐藏层的通道数
        :param filter_channels: 前馈网络的过滤通道数
        :param n_heads: 注意力头数
        :param n_layers: 编码器层数
        :param kernel_size: 卷积核大小
        :param p_dropout: Dropout的概率
        """
        super().__init__()
        self.hidden_channels = hidden_channels  # 隐藏层通道数
        self.filter_channels = filter_channels  # 过滤通道数
        self.n_heads = n_heads  # 注意力头数
        self.n_layers = n_layers  # 编码器层数
        self.kernel_size = kernel_size  # 卷积核大小
        self.p_dropout = p_dropout  # Dropout概率

        # Dropout层
        self.drop = torch.nn.Dropout(p_dropout)

        # 初始化注意力层、前馈网络层和归一化层
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        
        # 初始化每一层
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))  # 第一层归一化
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))  # 第二层归一化

    def forward(self, x, x_mask):
        """
        前向传播方法，依次通过每一层注意力和前馈神经网络。
        :param x: 输入张量
        :param x_mask: 掩码张量，用于忽略无效位置
        :return: 输出张量
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # 创建注意力掩码
        for i in range(self.n_layers):
            x = x * x_mask  # 应用输入掩码
            y = self.attn_layers[i](x, x, attn_mask)  # 通过注意力层
            y = self.drop(y)  # 应用Dropout
            x = self.norm_layers_1[i](x + y)  # 加法残差和归一化
            y = self.ffn_layers[i](x, x_mask)  # 通过前馈网络层
            y = self.drop(y)  # 应用Dropout
            x = self.norm_layers_2[i](x + y)  # 加法残差和归一化
        x = x * x_mask  # 返回掩码后的输出
        return x

class TextEncoder(nn.Module):
    """
    TextEncoder类实现了一个文本编码器，该编码器通常用于文本到特征的映射，包含词嵌入、预处理、编码器和时长预测器等部分。
    该编码器可以根据需要处理多种说话人并预测音频合成中的时长。
    """

    def __init__(
        self,
        encoder_type,
        encoder_params,
        duration_predictor_params,
        n_vocab,
        n_spks=1,
        spk_emb_dim=128,
    ):
        """
        初始化TextEncoder类。
        :param encoder_type: 编码器类型（例如：Transformer）
        :param encoder_params: 编码器参数，包括隐藏层通道数、头数等
        :param duration_predictor_params: 时长预测器的参数
        :param n_vocab: 词汇表大小
        :param n_spks: 说话人的数量（默认值为1）
        :param spk_emb_dim: 说话人嵌入的维度（默认值为128）
        """
        super().__init__()
        self.encoder_type = encoder_type  # 编码器类型
        self.n_vocab = n_vocab  # 词汇表大小
        self.n_feats = encoder_params.n_feats  # 特征维度
        self.n_channels = encoder_params.n_channels  # 编码器通道数
        self.spk_emb_dim = spk_emb_dim  # 说话人嵌入的维度
        self.n_spks = n_spks  # 说话人数量

        # 初始化词嵌入层
        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)  # 初始化权重

        # 初始化预处理层
        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.n_channels,
                self.n_channels,
                self.n_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x  # 如果没有预处理，则直接返回输入

        # 初始化编码器
        self.encoder = Encoder(
            encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )

        # 初始化输出卷积层和时长预测器
        self.proj_m = torch.nn.Conv1d(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), self.n_feats, 1)
        self.proj_w = DurationPredictor(
            self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, x, x_lengths, spks=None):
        """
        前向传播方法，运行文本输入通过编码器和时长预测器。
        
        参数:
            x (torch.Tensor): 文本输入，形状为 (batch_size, max_text_length)
            x_lengths (torch.Tensor): 文本长度，形状为 (batch_size,)
            spks (torch.Tensor, 可选): 说话人ID，形状为 (batch_size,)
        
        返回:
            mu (torch.Tensor): 编码器的输出，形状为 (batch_size, n_feats, max_text_length)
            logw (torch.Tensor): 时长预测的对数值，形状为 (batch_size, 1, max_text_length)
            x_mask (torch.Tensor): 输入的掩码，形状为 (batch_size, 1, max_text_length)
        """
        # 词嵌入并放大以适配通道数
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)  # 转置形状，以便后续操作

        # 计算文本掩码
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # 通过预处理层
        x = self.prenet(x, x_mask)
        
        # 如果有多个说话人，拼接说话人嵌入
        if self.n_spks > 1:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)

        # 通过编码器
        x = self.encoder(x, x_mask)
        
        # 通过输出卷积层
        mu = self.proj_m(x) * x_mask

        # 通过时长预测器
        x_dp = torch.detach(x)  # 在计算时长时不需要梯度
        logw = self.proj_w(x_dp, x_mask)

        return mu, logw, x_mask
