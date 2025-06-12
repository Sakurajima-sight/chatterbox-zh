# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import torch

'''
def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=torch.bool)
    return torch.tril(ret)
'''


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """创建带有块大小的后续步骤掩码 (size, size)，用于流式编码器。

    该掩码主要用于流式编码器，在流式处理时通过块大小控制当前块的可见区域。

    参数:
        size (int): 掩码的大小
        chunk_size (int): 块的大小
        num_left_chunks (int): 剩余的块数量
            <0: 使用完整块
            >=0: 使用 num_left_chunks
        device (torch.device): "cpu" 或 "cuda" 或 torch.Tensor.device

    返回:
        torch.Tensor: 掩码

    示例:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    # 说明：该实现经过修改，符合ONNX导出要求，但实际上并不支持num_left_chunks。
    # 在我们实现推理缓存后，这部分将不再需要，未来会去除。
    
    # 生成位置索引
    pos_idx = torch.arange(size, device=device)
    # 计算每个位置所属的块的最大值
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    # 创建掩码，表示每个位置是否属于其所在块的有效区域
    ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    return ret


def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """为编码器应用可选的块掩码。

    该函数主要用于动态和静态块大小的掩码应用，适用于流式推理和训练。

    参数:
        xs (torch.Tensor): 填充后的输入，形状为 (B, L, D)，L 为最大长度
        masks (torch.Tensor): 输入的掩码，形状为 (B, 1, L)
        use_dynamic_chunk (bool): 是否使用动态块大小
        use_dynamic_left_chunk (bool): 是否在训练中使用动态的剩余块
        decoding_chunk_size (int): 动态块的大小，当为 0 时表示训练时使用随机动态块；
            <0 时，表示解码时使用完整块；>0 时，表示解码时使用固定的块大小。
        static_chunk_size (int): 静态块大小，用于静态块训练或解码，如果大于 0 且 use_dynamic_chunk 为真，则该参数将被忽略。
        num_decoding_left_chunks (int): 解码时剩余块的数量，块大小由 decoding_chunk_size 指定。
            >=0 时，使用 num_decoding_left_chunks
            <0 时，使用所有剩余的块。
        enable_full_context (bool): 
            True：块大小为 [1, 25] 或完整上下文（最大长度）
            False：块大小为 U[1, 25] 的均匀分布。

    返回:
        torch.Tensor: 输入 `xs` 的块掩码
    """
    # 是否使用块掩码
    if use_dynamic_chunk:
        max_len = xs.size(1)  # 获取最大长度
        if decoding_chunk_size < 0:
            # 解码时使用完整块
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            # 解码时使用固定块大小
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # 动态块大小，块大小范围为 [1, 25] 或完整上下文（最大长度）
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1, )).item()

        # 使用 subsequent_chunk_mask 函数生成块掩码
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size, num_left_chunks, xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        # 使用静态块大小
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size, num_left_chunks, xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        # 不使用块掩码，直接返回原始掩码
        chunk_masks = masks

    # 确保块掩码的数据类型为 bool
    assert chunk_masks.dtype == torch.bool

    # 检查是否有时间步的掩码全为 False，如果有则强制设置为 True
    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        logging.warning('某些时间步的块掩码全为 False，强制设置为 True，请确保它们在未来的计算中被掩蔽！')
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """生成包含填充部分索引的掩码张量。

    该掩码用于标记输入中的填充部分。

    参数:
        lengths (torch.Tensor): 每个样本的长度，形状为 (B,)
        max_len (int): 最大长度，默认值为 0，表示使用输入序列的最大长度。

    返回:
        torch.Tensor: 掩码张量，包含填充部分的索引。

    示例:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)  # 获取批次大小
    max_len = max_len if max_len > 0 else lengths.max().item()  # 计算最大长度
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)  # 生成序列范围
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)  # 扩展序列范围以匹配批次大小
    seq_length_expand = lengths.unsqueeze(-1)  # 扩展长度为列向量
    mask = seq_range_expand >= seq_length_expand  # 生成掩码，填充部分为 True
    return mask
