from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import (
    GEGLU,
    GELU,
    AdaLayerNorm,
    AdaLayerNormZero,
    ApproximateGELU,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.utils.torch_utils import maybe_allow_in_graph


class SnakeBeta(nn.Module):
    """
    一个修改版的Snake激活函数，使用独立的参数控制周期分量的幅度。
    形状：
        - 输入： (B, C, T)
        - 输出： (B, C, T)，与输入形状相同
    参数：
        - alpha - 可训练参数，控制频率
        - beta - 可训练参数，控制幅度
    参考文献：
        - 该激活函数基于Liu Ziyin, Tilman Hartwig, Masahito Ueda的论文进行修改： https://arxiv.org/abs/2006.08195
    示例：
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        初始化。
        输入：
            - in_features: 输入的形状
            - alpha - 可训练参数，控制频率
            - beta - 可训练参数，控制幅度
            alpha默认初始化为1，较高的值意味着较高的频率。
            beta默认初始化为1，较高的值意味着较大的幅度。
            alpha将与模型的其他部分一起进行训练。
        """
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # 初始化 alpha 和 beta
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # 如果是log scale，初始化为零
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # 如果是线性尺度，初始化为1
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        函数的前向传播方法。
        对输入逐元素应用该函数。
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        # 应用 SnakeBeta 激活函数
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class FeedForward(nn.Module):
    r"""
    前馈神经网络层。

    参数：
        dim (`int`): 输入通道的数量。
        dim_out (`int`, *可选*): 输出通道的数量。如果没有给出，默认为`dim`。
        mult (`int`, *可选*, 默认为4): 用于隐藏维度的乘数。
        dropout (`float`, *可选*, 默认为0.0): 使用的dropout概率。
        activation_fn (`str`, *可选*, 默认为 `"geglu"`): 前馈网络中使用的激活函数。
        final_dropout (`bool` *可选*, 默认为False): 是否在最后应用dropout。
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # 根据指定的激活函数选择相应的激活层
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        elif activation_fn == "snakebeta":
            act_fn = SnakeBeta(dim, inner_dim)

        self.net = nn.ModuleList([])
        # 输入层投影
        self.net.append(act_fn)
        # Dropout层
        self.net.append(nn.Dropout(dropout))
        # 输出层投影
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        # 如果需要，最后的dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        # 顺序执行网络中的每个模块
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    基本的Transformer模块。该模块包含自注意力、交叉注意力（可选）以及前馈网络部分，常用于Transformer架构中。

    参数：
        dim (`int`): 输入和输出的通道数。
        num_attention_heads (`int`): 多头注意力中的头数。
        attention_head_dim (`int`): 每个头的通道数。
        dropout (`float`, *可选*, 默认为0.0): 使用的dropout概率。
        cross_attention_dim (`int`, *可选*): 用于交叉注意力的`encoder_hidden_states`向量大小。
        only_cross_attention (`bool`, *可选*):
            是否仅使用交叉注意力层。在这种情况下，使用两个交叉注意力层。
        double_self_attention (`bool`, *可选*):
            是否使用两个自注意力层。在这种情况下，不使用交叉注意力层。
        activation_fn (`str`, *可选*, 默认为 `"geglu"`): 前馈网络中使用的激活函数。
        num_embeds_ada_norm (`int`, *可选*): 训练过程中使用的扩散步骤数。请参阅 `Transformer2DModel`。
        attention_bias (`bool`, *可选*, 默认为 `False`): 配置注意力是否应包含偏置参数。
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        """
        初始化BasicTransformerBlock类。
        参数解释见上方注释。
        """
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # 如果使用AdaLayerNorm，需要num_embeds_ada_norm
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        # 如果选择了AdaLayerNorm但没有定义num_embeds_ada_norm，则抛出错误
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # 定义三个块，每个块有自己的归一化层。
        # 1. 自注意力
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. 交叉注意力
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. 前馈网络
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # 设置chunk的前馈
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        """
        前向传播过程。
        包括：
        1. 自注意力
        2. 交叉注意力（如果有）
        3. 前馈神经网络
        """
        # 1. 自注意力
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. 交叉注意力
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. 前馈神经网络
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # 使用"feed_forward_chunk_size"来节省内存
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states
