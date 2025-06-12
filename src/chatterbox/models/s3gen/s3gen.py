# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
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

import logging
import numpy as np
import torch
import torchaudio as ta
from functools import lru_cache
from typing import Optional
from omegaconf import DictConfig

from ..s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .xvector import CAMPPlus
from .utils.mel import mel_spectrogram
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder


def drop_invalid_tokens(x):
    """
    Function to filter out invalid tokens from input tensor.
    This function ensures that only valid speech tokens are retained, removing any tokens that exceed the 
    speech vocabulary size (SPEECH_VOCAB_SIZE).

    参数:
    - x: 输入张量，应该是一个形状为 (batch_size, sequence_length) 的张量
    
    返回:
    - 只保留有效token的张量
    """
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    """
    Function to return a resampler for transforming audio from one sample rate to another.
    
    参数:
    - src_sr: 源采样率
    - dst_sr: 目标采样率
    - device: 用于计算的设备，例如 CPU 或 GPU
    
    返回:
    - 返回一个用于重采样的 Resample 对象
    """
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(torch.nn.Module):
    """
    CosyVoice2的CFM解码器将S3语音token映射到mel频谱图。

    TODO: 使这些模块可配置？
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")  # 使用指定的tokenizer
        self.mel_extractor = mel_spectrogram  # 提取mel频谱图（TODO：可以将其做成一个torch模块？）
        self.speaker_encoder = CAMPPlus()  # 使用默认参数初始化CAMPPlus说话人编码器

        # 初始化上采样的Conformer编码器
        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        # 初始化条件解码器
        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
        )

        # CFM的参数配置
        cfm_params = DictConfig({
            "sigma_min": 1e-06,
            "solver": 'euler',
            "t_scheduler": 'cosine',
            "training_cfg_rate": 0.2,
            "inference_cfg_rate": 0.7,
            "reg_loss_type": 'l1',
        })

        # 初始化条件CFM解码器
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        # 初始化CausalMaskedDiffWithXvec流
        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder
        )

        self.resamplers = {}

    @property
    def device(self):
        """
        获取当前模型所在的设备（GPU/CPU）。
        
        返回：
        - 设备信息
        """
        params = self.tokenizer.parameters()
        return next(params).device

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        """
        从参考波形提取特征（包括mel频谱图、说话人嵌入和token化的参考音频），并将其准备好用于生成。

        参数：
        - ref_wav: 参考波形（torch.Tensor 类型）
        - ref_sr: 参考波形的采样率
        - device: 设备（CPU或GPU），默认为"auto"，会自动选择设备
        - ref_fade_out: 是否应用淡出处理，默认为True

        返回：
        - 返回一个字典，包含参考音频的token、mel频谱图、说话人嵌入等信息
        """
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)

        if ref_wav.size(1) > 10 * ref_sr:
            print("WARNING: cosydec received ref longer than 10s")

        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)

        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(device)
        ref_mels_24_len = None

        # 将参考波形重采样到16kHz
        ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav).to(device)

        # 获取说话人的嵌入
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)

        # 对16kHz的参考音频进行token化
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16)

        # 确保mel频谱图的长度是2倍的token长度
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
        )

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        """
        从S3语音tokens和参考波形生成波形，其中参考波形的说话人音色将被推断出来。

        注意：
        - 说话人编码器接受16 kHz的波形输入。
        - S3TokenizerV2接受16 kHz的波形输入。
        - 参考波形的mel频谱图假定为24 kHz输入信号。
        - 此函数设计为仅支持batch_size=1。

        参数：
        - `speech_tokens`: S3语音tokens，形状为 [B=1, T]
        - `ref_wav`: 参考波形 (`torch.Tensor` 类型，形状为 [B=1, T])
        - `ref_sr`: 参考采样率
        - `finalize`: 是否已完成流式处理。如果为False，将忽略最后3个tokens。

        返回：
        - 生成的mel频谱图
        """
        assert (ref_wav is None) ^ (ref_dict is None), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            # 类型和设备转换（如果是prod API调用，这里会是numpy类型）
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(self.device)

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        # assert speech_tokens.shape[0] == 1, "only batch size of one allowed for now"
        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)

        # 通过流进行推理，生成mel频谱图
        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            **ref_dict,
        )
        return output_mels


class S3Token2Wav(S3Token2Mel):
    """
    CosyVoice2的解码器是由token-to-mel（CFM）和mel-to-waveform（HiFiGAN）模块组成的拼接结构。

    TODO: 使这些模块可配置？
    """

    def __init__(self):
        super().__init__()

        # 初始化F0预测器（基于卷积RNN）
        f0_predictor = ConvRNNF0Predictor()

        # 初始化HiFiGAN（用于从mel频谱图生成波形）
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # 为了减少伪影（artifacts），我们对音频进行修剪和淡入处理
        n_trim = S3GEN_SR // 50  # 20ms = 一个帧的半个时间
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)  # 注册为buffer，自动进行设备转换

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False
    ):
        """
        将S3语音tokens转换为波形，参考波形的说话人音色被推断出来。

        参数：
        - `speech_tokens`: S3语音tokens
        - `ref_wav`: 参考波形（用于提取说话人嵌入和音色）
        - `ref_sr`: 参考波形的采样率
        - `ref_dict`: 预计算的参考嵌入（从生产API获取）
        - `finalize`: 是否为流式处理的最后一步

        返回：
        - 生成的波形
        """
        # 先通过父类的forward方法生成mel频谱图
        output_mels = super().forward(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)

        # TODO jrm: 暂时忽略速度控制（mel插值）和HiFTGAN缓存机制
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        # 通过HiFiGAN生成波形
        output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, cache_source=hift_cache_source)

        if not self.training:
            # 训练时不进行此处理，测试时减少参考片段的"溢出"
            output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        """
        在推理模式下执行流的推理，生成mel频谱图。

        参数：
        - `speech_tokens`: 输入的S3语音tokens
        - `ref_wav`: 参考波形
        - `ref_sr`: 参考波形的采样率
        - `ref_dict`: 预计算的参考嵌入
        - `finalize`: 是否为最后一步处理

        返回：
        - mel频谱图
        """
        return super().forward(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        """
        在推理模式下，通过HiFiGAN从mel频谱图生成波形。

        参数：
        - `speech_feat`: 输入的mel频谱图
        - `cache_source`: 缓存源（用于流式处理）

        返回：
        - 生成的波形
        """
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(self.device)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        cache_source: torch.Tensor = None,  # NOTE: this arg is for streaming, it can probably be removed here
        finalize: bool = True,
    ):
        """
        在推理模式下，通过S3语音tokens生成波形，参考波形的说话人音色会被推断出来。

        参数：
        - `speech_tokens`: 输入的S3语音tokens
        - `ref_wav`: 参考波形
        - `ref_sr`: 参考波形的采样率
        - `ref_dict`: 预计算的参考嵌入
        - `cache_source`: 用于流式处理的缓存源
        - `finalize`: 是否为最后一步处理

        返回：
        - 生成的波形
        - 生成的源数据
        """
        # 先生成mel频谱图
        output_mels = self.flow_inference(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)
        
        # 再从mel频谱图生成波形
        output_wavs, output_sources = self.hift_inference(output_mels, cache_source)

        # 测试时减少参考片段的"溢出"
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources
