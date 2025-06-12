# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass
from types import MethodType

# 设置日志记录器
logger = logging.getLogger(__name__)

@dataclass
class AlignmentAnalysisResult:
    """
    对齐分析结果的数据结构
    该类包含了用于分析对齐的各类标志及其相关数据
    """
    # 是否检测到该帧为噪声开始部分，可能是幻觉产生的部分
    false_start: bool
    # 是否检测到该帧为长尾部分，可能是幻觉产生的部分
    long_tail: bool
    # 是否检测到该帧为重复的文本内容
    repetition: bool
    # 是否检测到该帧的对齐位置与前一帧偏差过大
    discontinuity: bool
    # 推理是否已到达文本的结束标记（EOS），如果推理提前停止则为False
    complete: bool
    # 文本token序列的近似位置。可用于生成在线时间戳。
    position: int


class AlignmentStreamAnalyzer:
    """
    对齐流分析器类
    该类用于分析模型中对齐的情况，主要用于基于Transformer模型进行流式生成的推理检测，
    它通过注意力机制分析每一帧与文本的对齐情况，包括是否存在重复、幻觉、或长尾现象等。
    """
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        """
        初始化 AlignmentStreamAnalyzer 类的实例。
        
        :param tfmr: Transformer 模型实例
        :param queue: 数据队列
        :param text_tokens_slice: 文本 tokens 的切片（元组，i, j），表示文本对齐的范围
        :param alignment_layer_idx: 对齐分析所使用的注意力层索引
        :param eos_idx: EOS（结束）token的索引
        """
        # 初始化参数
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        # 将 hook 添加到指定的对齐层（即自注意力层）
        self.last_aligned_attn = None
        self._add_attention_spy(tfmr, alignment_layer_idx)

    def _add_attention_spy(self, tfmr, alignment_layer_idx):
        """
        向指定的对齐层添加一个 hook，用于收集注意力输出。
        由于 `output_attentions=True` 与优化的注意力内核不兼容，因此只对一个层使用它，避免性能下降。
        
        :param tfmr: Transformer 模型实例
        :param alignment_layer_idx: 用于对齐分析的注意力层索引
        """
        def attention_forward_hook(module, input, output):
            """
            注意力前向 hook 函数，获取注意力矩阵的输出。
            
            :param module: 当前的模块（注意力层）
            :param input: 输入数据
            :param output: 输出数据
            """
            step_attention = output[1].cpu()  # 获取注意力权重 (B, 16, N, N)
            self.last_aligned_attn = step_attention[0].mean(0)  # 取第一批次的均值作为对齐注意力矩阵

        # 注册 hook 到目标层
        target_layer = tfmr.layers[alignment_layer_idx].self_attn
        hook_handle = target_layer.register_forward_hook(attention_forward_hook)

        # 备份原始的 forward 方法
        original_forward = target_layer.forward
        def patched_forward(self, *args, **kwargs):
            kwargs['output_attentions'] = True
            return original_forward(*args, **kwargs)

        # 替换 forward 方法
        target_layer.forward = MethodType(patched_forward, target_layer)

    def step(self, logits):
        """
        执行一步对齐分析，并根据需要修改 logits 强制 EOS（结束）token。
        
        :param logits: 当前的预测 logits
        :return: 修改后的 logits
        """
        # 获取当前对齐矩阵的 chunk（每次处理一帧）
        aligned_attn = self.last_aligned_attn  # 对齐注意力矩阵 (N, N)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # 第一帧包含条件信息、文本 token 和 BOS token
            A_chunk = aligned_attn[j:, i:j].clone().cpu()  # (T, S)
        else:
            # 后续帧由于使用 KV-caching，每次只有 1 帧
            A_chunk = aligned_attn[:, i:j].clone().cpu()  # (1, S)

        # 对齐矩阵的处理
        A_chunk[:, self.curr_frame_pos + 1:] = 0  # 避免过度跨越的部分影响

        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # 更新文本位置
        cur_text_posn = A_chunk[-1].argmax()  # 当前的文本位置
        discontinuity = not(-4 < cur_text_posn - self.text_position < 7)  # 检查是否存在大幅度跳跃
        if not discontinuity:
            self.text_position = cur_text_posn

        # 处理假起始：检测到幻觉行为时，返回 False Start
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # 检查推理是否完成
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # 计算最后 token 持续时间，用于判定是否为长尾幻觉
        last_text_token_duration = A[15:, -3:].sum()

        # 检测幻觉长尾
        long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 10)  # 长尾检测（超过 400ms）

        # 检查是否有重复生成
        repetition = self.complete and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        # 如果检测到长尾或重复，则强制发出 EOS token
        if long_tail or repetition:
            logger.warn(f"forcing EOS token, {long_tail=}, {repetition=}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        # 防止提前终止，抑制 EOS token
        if cur_text_posn < S - 3:  # FIXME: 这里的阈值值可以调整
            logits[..., self.eos_idx] = -2**15

        self.curr_frame_pos += 1
        return logits
