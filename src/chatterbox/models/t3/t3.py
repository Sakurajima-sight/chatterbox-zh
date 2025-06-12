# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor

from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer


logger = logging.getLogger(__name__)

# 创建一个字典类，支持通过属性访问字典项
class AttrDict(dict):
    """ 
    这是一个扩展了字典的类，允许通过属性访问字典的键值对。 
    例如，a = AttrDict({'key': 'value'})，可以通过 a.key 访问 'value'。
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  # 将字典的__dict__指向自身，允许通过属性访问


# 确保文本序列中包含开始和结束标记
def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    """
    检查文本标记序列中是否包含开始（start_text_token）和结束（stop_text_token）标记。
    如果缺少任意一个标记，会抛出一个断言错误。
    
    参数:
    text_tokens (Tensor): 输入的文本标记序列
    hp: 配置，包含 start_text_token 和 stop_text_token
    
    返回:
    None
    """
    B = text_tokens.size(0)  # 获取批次大小
    # 检查是否包含开始标记
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    # 检查是否包含结束标记
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS 模型，使用 Huggingface transformer 模型作为主干网络
        * 分词，包括开始 / 结束标记，始终在此类之外进行添加
        * 条件数据（如 CLAP、情感等）在单独的文件中，以提高模块化程度
        * 注意！此类假设使用相对位置编码，如果使用绝对位置编码，至少需要在语音标记开始时重置位置为 0，
            并且可以选择为语音使用不同的 PE 嵌入空间。
    """

    def __init__(self, hp=T3Config()):
        """
        初始化 T3 模型，配置包括模型参数、Transformer 配置等。

        参数:
        hp (T3Config): 配置对象，包含模型的超参数
        """
        super().__init__()
        self.hp = hp
        # 根据配置加载 Llama 配置
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = LlamaModel(self.cfg)  # 初始化 LlamaModel
        self.dim = self.cfg.hidden_size  # 获取 Llama 模型的隐藏层维度
        self.deepspeed_patch_applied = False  # 深度学习优化补丁标记

        # 条件嵌入（例如 CLAP、情感数据等）
        self.cond_enc = T3CondEnc(hp)  # 条件编码器
        # 文本和语音的嵌入层
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # 自定义位置嵌入
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # Logit 投影层
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False

    @property
    def device(self):
        """
        获取当前设备，主要用于在模型中处理数据时获取模型参数所在设备。
        """
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        准备条件数据，将条件提示的语音标记转换为嵌入。
        这个函数会嵌入条件数据，确保它适合用于后续模型的计算。
        
        参数:
        t3_cond (T3Cond): 包含条件数据（如语音标记）的对象
        
        返回:
        条件嵌入（形状为 (B, len_cond, dim)）
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        """
        准备输入的嵌入（跳过主干 Transformer 的嵌入层）。
        
        参数:
        t3_cond (T3Cond): 条件数据
        text_tokens (LongTensor): 文本标记
        speech_tokens (LongTensor): 语音标记
        cfg_weight (float): 控制条件生成的权重
        
        返回:
        输入嵌入（包含条件、文本和语音嵌入的拼接）
        """
        # 准备条件嵌入
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            text_emb[1].zero_()  # CFG 无条件生成

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        # 确保条件嵌入与文本嵌入批次大小一致
        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # 拼接条件嵌入、文本嵌入和语音嵌入
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        """
        T3 模型的前向传播，处理输入文本和语音标记，生成文本和语音的 logits 输出。
        
        参数:
        t3_cond (T3Cond): 条件数据
        text_tokens (LongTensor): 文本标记
        text_token_lens (LongTensor): 文本标记的长度
        speech_tokens (LongTensor): 语音标记
        speech_token_lens (LongTensor): 语音标记的长度
        training (bool): 是否处于训练模式
        
        返回:
        输出包含文本和语音的 logits 及潜在表示
        """
        _ensure_BOT_EOT(text_tokens, self.hp)  # 确保文本中包含开始和结束标记

        # 准备输入嵌入
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # 主干 Transformer 前向传播
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # 获取最后一层的输出 (B, seq, dim)

        # 后处理：从 hidden states 中分离文本和语音部分
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # Logit 投影
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss_old(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        """
        训练方法，用于计算文本和语音的交叉熵损失。
        
        参数:
        t3_cond (T3Cond): 条件数据
        text_tokens (LongTensor): 文本标记
        text_token_lens (LongTensor): 文本标记的长度
        speech_tokens (LongTensor): 语音标记
        speech_token_lens (LongTensor): 语音标记的长度
        
        返回:
        loss_text (Tensor): 文本的交叉熵损失
        loss_speech (Tensor): 语音的交叉熵损失
        """
        len_text = text_tokens.size(1)  # 获取文本标记的长度
        len_speech = speech_tokens.size(1)  # 获取语音标记的长度
        
        assert len_text == text_token_lens.max()  # 确保文本标记长度一致
        assert len_speech == speech_token_lens.max()  # 确保语音标记长度一致

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # 调用前向传播获取模型输出

        # 计算交叉熵损失
        IGNORE_ID = -100  # 忽略的标记ID
        device = out.text_logits.device  # 获取设备信息
        # 创建文本和语音的掩码
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        # 掩码处理文本和语音标记
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)

        # 计算文本和语音的交叉熵损失
        loss_text = F.cross_entropy(out.text_logits.transpose(1, 2), masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out.speech_logits.transpose(1, 2), masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech  # 返回损失值

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,        # (B, S_text_padded), 包含 BOS 和 EOS 标记
        text_token_lens: torch.LongTensor,    # (B,), 实际长度，包括 BOS 和 EOS
        speech_tokens: torch.LongTensor,      # (B, S_speech_padded), 包含 BOS 和 EOS 标记
        speech_token_lens: torch.LongTensor,  # (B,), 实际长度，包括 BOS 和 EOS
        labels_text: torch.LongTensor,        # (B, S_text_padded-1), 已经用 -100 掩码
        labels_speech: torch.LongTensor       # (B, S_speech_padded-1), 已经用 -100 掩码
    ):
        """
        使用来自数据整理器的预掩码标签计算文本和语音的交叉熵。
        假设:
        - labels_text[t] 对应于预测 text_tokens[:, 1:]，已用 -100 掩码忽略部分
        - labels_speech[t] 对应于预测 speech_tokens[:, 1:]，已用 -100 掩码忽略部分
        """

        # 1) 运行模型获取 logits
        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )
        # out.text_logits: (B, S_text_padded, V_text)
        # out.speech_logits: (B, S_speech_padded, V_speech)
        device = out.text_logits.device  # 获取设备信息
        IGNORE_ID = -100  # 忽略的标记ID

        # --- 计算文本损失 (直接使用 labels_text) ---
        logits_for_text = out.text_logits[:, :-1, :].contiguous()  # (B, S_text_padded-1, V_text)
        # labels_text 已经是形状 (B, S_text_padded-1)，并且用 -100 掩码忽略部分
        if logits_for_text.size(1) == 0:
            loss_text = torch.tensor(0.0, device=device, requires_grad=self.training)
        else:
            loss_text = F.cross_entropy(
                logits_for_text.transpose(1, 2),  # (B, V_text, S_text_padded-1)
                labels_text,                      # (B, S_text_padded-1)，ignore_index=–100
                ignore_index=IGNORE_ID
            )

        # --- 计算语音损失 (直接使用 labels_speech) ---
        logits_for_speech = out.speech_logits[:, :-1, :].contiguous()  # (B, S_speech_padded-1, V_speech)
        # labels_speech 已经是形状 (B, S_speech_padded-1)，并且用 -100 掩码忽略部分
        if logits_for_speech.size(1) == 0:
            loss_speech = torch.tensor(0.0, device=device, requires_grad=self.training)
        else:
            loss_speech = F.cross_entropy(
                logits_for_speech.transpose(1, 2),  # (B, V_speech, S_speech_padded-1)
                labels_speech,                      # (B, S_speech_padded-1)，ignore_index=–100
                ignore_index=IGNORE_ID
            )

        return loss_text, loss_speech, out.speech_logits  # 返回文本损失、语音损失和语音 logits

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor]=None,

        # 其他条件数据
        prepend_prompt_speech_tokens: Optional[Tensor]=None,

        # HF 生成参数
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        length_penalty=1.0,
        repetition_penalty=2.0,
        cfg_weight=0,
    ):
        """
        推理方法，用于生成语音标记。基于输入的文本标记，生成语音序列。
        
        参数:
        t3_cond (T3Cond): 条件数据
        text_tokens (Tensor): 输入的文本标记，可以是 1D（无批次）或 2D（有批次）张量
        initial_speech_tokens (Optional[Tensor]): 初始语音标记，默认为 None，则会使用开始标记
        prepend_prompt_speech_tokens (Optional[Tensor]): 用于前置的语音标记（未实现）
        
        其他生成参数：
        num_return_sequences (int): 要生成的序列数量
        max_new_tokens (Optional[int]): 最大生成的新标记数
        stop_on_eos (bool): 是否在遇到 EOS 标记时停止生成
        do_sample (bool): 是否进行采样
        temperature (float): 温度参数，用于调整采样概率
        top_p (float): 核采样的 p 值
        length_penalty (float): 长度惩罚
        repetition_penalty (float): 重复惩罚
        cfg_weight (float): 条件生成的权重
        """
        
        # 输入校验 / 清理
        assert prepend_prompt_speech_tokens is None, "未实现的功能"
        _ensure_BOT_EOT(text_tokens, self.hp)  # 确保文本中包含开始和结束标记
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # 默认初始语音为单个开始标记
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # 准备自定义输入嵌入
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # 为了使用标准的 Huggingface 生成方法，我们需要扩展一些方法来注入自定义逻辑
        self.compiled = False

        # 如果模型未编译，进行模型编译
        if not self.compiled:
            alignment_stream_analyzer = AlignmentStreamAnalyzer(
                self.tfmr,
                None,
                text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                alignment_layer_idx=9,  # TODO: 可以作为超参数设置
                eos_idx=self.hp.stop_speech_token,
            )
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        device = embeds.device

        # 使用开始标记进行初始化
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # 形状: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # 由于条件生成的需要，BATCH_SIZE=2
        bos_embed = torch.cat([bos_embed, bos_embed])

        # 如果 cfg_weight > 0，将条件和 BOS 标记结合
        if cfg_weight > 0:
            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        else:
            inputs_embeds = embeds

        # 生成标记 id 的追踪，初始化为 BOS 标记。
        generated_ids = bos_token.clone()
        predicted = []  # 用于存储预测的标记

        # 实例化 Logits 处理器
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        # ---- 初始前向传播（没有 kv_cache）----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # 初始化 kv_cache 为完整的上下文。
        past = output.past_key_values

        # ---- 生成循环，使用 kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits = output.logits[:, -1, :]

            # CFG 调整
            if cfg_weight > 0.0:
                logits_cond = logits[0:1]
                logits_uncond = logits[1:2]
                logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

            logits = logits.squeeze(1)

            # 应用温度缩放
            if temperature != 1.0:
                logits = logits / temperature

            # 应用重复惩罚和 top-p 过滤
            logits = repetition_penalty_processor(generated_ids, logits)
            logits = top_p_warper(None, logits)

            # 将 logits 转化为概率并采样下一个标记
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # 形状: (B, 1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # 检查是否遇到 EOS 标记
            if next_token.view(-1) == self.hp.stop_speech_token:
                break

            # 获取新标记的嵌入
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            # 对于 CFG
            if cfg_weight > 0.0:
                next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # 使用新标记和缓存的 past 进行前向传播
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            # 更新 kv_cache。
            past = output.past_key_values

        # 拼接所有预测的标记
        predicted_tokens = torch.cat(predicted, dim=1)  # 形状: (B, num_tokens)
        return predicted_tokens  # 返回预测的标记
