from typing import Optional

import torch
from torch import nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaPreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class T3HuggingfaceBackend(LlamaPreTrainedModel, GenerationMixin):
    """
    用于重载 HuggingFace 接口方法，以便使用标准的 `generate` 方法，并支持自定义嵌入层 / 逻辑层。
    
    注意：需要继承 "*PreTrainedModel" 以避免重新初始化权重！
    """
    def __init__(
        self,
        config: LlamaConfig,
        llama: LlamaModel,
        *,
        speech_enc,  # 自定义语音编码层
        speech_head,  # 自定义语音头部（logit 投影层）
        latents_queue=None,  # 潜在队列
        logits_queue=None,  # logits 队列
        alignment_stream_analyzer: 'AlignmentStreamAnalyzer' = None,  # 对齐流分析器
    ):
        """
        初始化 T3HuggingfaceBackend 类的实例。
        
        :param config: Llama 配置
        :param llama: Llama 模型实例
        :param speech_enc: 语音编码层，用于将文本 token 嵌入转换为语音嵌入
        :param speech_head: 语音头部，用于语音 logit 的投影
        :param latents_queue: 潜在队列
        :param logits_queue: logits 队列
        :param alignment_stream_analyzer: 对齐流分析器（用于处理流式生成中的对齐情况）
        """
        super().__init__(config)
        self.model = llama  # Llama 模型
        self.speech_enc = speech_enc  # 语音编码层
        self.speech_head = speech_head  # 语音头部
        self._added_cond = False  # 用于标记是否已经添加条件（即 decoder conditioning）
        self.alignment_stream_analyzer = alignment_stream_analyzer  # 对齐流分析器

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, decoder_cond: torch.Tensor, use_cache: bool, past_key_values=None,
        cache_position=None  # 在一些较新版本的 transformers 中新增的参数
    ):
        """
        这是 HuggingFace 的 `generate()` 方法调用时使用的方法。这里重载它以应用自定义的语音 token 嵌入层。
        
        :param input_ids: (B, S) 的 int64 类型张量，表示输入的 token 序列。
        :param decoder_cond: (B, T, C) 的 float32 类型张量，表示条件（前缀加到 <input_embeds> 上）
        :param use_cache: 是否使用缓存
        :param past_key_values: 缓存的键值对
        :param cache_position: 缓存位置（用于新版本的 transformers）
        
        :return: 返回包含 inputs_embeds 和其他需要的参数的字典
        """
        # 如果不使用缓存，只处理当前 token
        if not use_cache:
            past_key_values = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # 只取最后一个输入 token

        # 使用自定义的语音 token 嵌入层
        inputs_embeds = self.speech_enc(input_ids)

        # 如果需要，前缀 decoder 的条件
        if not self._added_cond:
            assert past_key_values is not None  # 应该是第一次调用
            if decoder_cond.size(0) != inputs_embeds.size(0):
                decoder_cond = decoder_cond.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([decoder_cond, inputs_embeds], dim=1)  # 将条件与输入嵌入连接起来
            self._added_cond = True

        return {
            "inputs_embeds": inputs_embeds,  # 返回修改后的嵌入
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @torch.inference_mode()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    ):
        """
        这是 HuggingFace 的 `generate()` 方法调用时使用的前向方法。重载这里的前向方法以应用我们的自定义层归一化和语音 logit 投影层。
        
        :param inputs_embeds: (B, S, C) 的 float32 类型张量，表示输入的嵌入。如果 past_key_values 被提供，S 应该是 1。
        :param past_key_values: 缓存的键值对
        :param use_cache: 是否使用缓存
        :param output_attentions: 是否输出注意力权重
        :param output_hidden_states: 是否输出隐藏层状态
        :param return_dict: 是否以字典形式返回输出
        
        :return: 返回 CausalLMOutputWithCrossAttentions 输出，包含 logits、past_key_values 等。
        """
        is_large_input = inputs_embeds.size(1) != 1
        has_cache = past_key_values is not None and len(past_key_values) > 0
        assert not (is_large_input and has_cache)  # 检查大输入与缓存条件冲突
        assert return_dict
        assert output_hidden_states

        # 通过 Llama 模型进行前向推理
        tfmr_out = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # 获取模型的最后一层隐藏状态
        hidden_states = tfmr_out.hidden_states[-1]  # (B, seq, dim)

        # 应用语音头部（logit 投影层）
        logits = self.speech_head(hidden_states)

        # 这里可以加上幻觉处理器，强制发出 EOS token
        # logits = self.alignment_stream_analyzer.step(logits)

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=tfmr_out.past_key_values,
            hidden_states=tfmr_out.hidden_states,
            attentions=tfmr_out.attentions,
        )
