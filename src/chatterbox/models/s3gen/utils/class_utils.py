# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
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

# 导入自定义激活函数
from ..transformer.activation import Swish
# 导入不同类型的子采样模块
from ..transformer.subsampling import (
    LinearNoSubsampling,  # 无子采样的线性层
    EmbedinigNoSubsampling,  # 无子采样的嵌入层
    Conv1dSubsampling2,  # 1D卷积子采样（步长为2）
    Conv2dSubsampling4,  # 2D卷积子采样（步长为4）
    Conv2dSubsampling6,  # 2D卷积子采样（步长为6）
    Conv2dSubsampling8,  # 2D卷积子采样（步长为8）
)
# 导入不同的定位编码方法
from ..transformer.embedding import (
    PositionalEncoding,  # 传统的位置信息编码
    RelPositionalEncoding,  # 相对位置编码
    WhisperPositionalEncoding,  # Whisper特定的位置信息编码
    LearnablePositionalEncoding,  # 可学习的位置信息编码
    NoPositionalEncoding  # 无位置信息编码
)
# 导入注意力机制
from ..transformer.attention import (
    MultiHeadedAttention,  # 多头自注意力
    RelPositionMultiHeadedAttention  # 相对位置多头自注意力
)
# 导入Espnet的相对位置编码
from ..transformer.embedding import EspnetRelPositionalEncoding
# 导入遗留的无子采样模块
from ..transformer.subsampling import LegacyLinearNoSubsampling


# 激活函数类映射：根据字符串名选择对应的激活函数类
COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,  # HardTanh激活函数
    "tanh": torch.nn.Tanh,  # Tanh激活函数
    "relu": torch.nn.ReLU,  # ReLU激活函数
    "selu": torch.nn.SELU,  # SELU激活函数
    "swish": getattr(torch.nn, "SiLU", Swish),  # Swish激活函数，SiLU是PyTorch的实现
    "gelu": torch.nn.GELU,  # GELU激活函数
}

# 子采样模块类映射：根据字符串名选择对应的子采样模块
COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,  # 无子采样的线性层
    "linear_legacy": LegacyLinearNoSubsampling,  # 遗留的无子采样线性层
    "embed": EmbedinigNoSubsampling,  # 无子采样的嵌入层
    "conv1d2": Conv1dSubsampling2,  # 1D卷积子采样（步长为2）
    "conv2d": Conv2dSubsampling4,  # 2D卷积子采样（步长为4）
    "conv2d6": Conv2dSubsampling6,  # 2D卷积子采样（步长为6）
    "conv2d8": Conv2dSubsampling8,  # 2D卷积子采样（步长为8）
    'paraformer_dummy': torch.nn.Identity  # Paraformer的dummy子采样（即不进行实际的操作）
}

# 位置信息编码类映射：根据字符串名选择对应的位置信息编码方法
COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,  # 传统的位置信息编码
    "abs_pos": PositionalEncoding,  # 绝对位置编码
    "rel_pos": RelPositionalEncoding,  # 相对位置编码
    "rel_pos_espnet": EspnetRelPositionalEncoding,  # Espnet风格的相对位置编码
    "no_pos": NoPositionalEncoding,  # 无位置编码
    "abs_pos_whisper": WhisperPositionalEncoding,  # Whisper风格的绝对位置编码
    "embed_learnable_pe": LearnablePositionalEncoding,  # 可学习的位置信息编码
}

# 注意力机制类映射：根据字符串名选择对应的注意力机制
COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,  # 普通的多头自注意力机制
    "rel_selfattn": RelPositionMultiHeadedAttention,  # 相对位置的多头自注意力机制
}
