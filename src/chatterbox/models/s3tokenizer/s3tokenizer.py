from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import (
    S3TokenizerV2,
    ModelConfig,
)

# S3TokenizerV2输入的采样率
S3_SR = 16_000
S3_HOP = 160  # 每秒100帧
S3_TOKEN_HOP = 640  # 每秒25个token
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 的子类，主要改动有：
    - 集成的 `forward` 方法
    - 在 `register_buffers` 中用 `_mel_filters` 和 `window` 计算 `log_mel_spectrogram`
    """
    # S3Tokenizer，继承自S3TokenizerV2，添加集成的forward方法和log mel谱计算
    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str="speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig()
    ):
        """
        构造函数，初始化mel滤波器和窗口等buffer。
        """
        super().__init__(name)

        self.n_fft = 400
        # 创建mel滤波器
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        # 创建hann窗口
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        """
        给定一组采样率相同的wavs，对其进行padding，使长度为40ms的整数倍（S3每秒25token）。
        """
        # 音频padding到40ms整数倍，便于后续分帧
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav,
                (0, intended_wav_len - wav.shape[-1]),
                mode="constant",
                value=0
            )
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        """  
        预处理音频，把numpy array转为tensor，维度对齐  
        """
        # 预处理输入音频，转换为Tensor且维度对齐
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        accelerator: 'Accelerator'=None,
        max_len: int=None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        前向推理，输入16kHz音频，输出token序列和长度
        参数
        ----
        - `wavs`: 16 kHz 采样率的语音音频
        - `max_len`：最大输出token序列长度（25 token/秒）。
        注意：如需更长序列，请对波形进行padding。
        """
        # 前向传播，提取mel特征并量化为token
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)
        if accelerator is None:
            tokenizer = self
        else:
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        计算log-Mel谱图（梅尔频谱），用于后续的tokenizer处理

        参数
        ----------
        audio: torch.Tensor, shape = (*)
            音频路径，或包含16kHz音频波形的NumPy数组或Tensor

        padding: int
            在右侧补零的样本数量

        返回
        -------
        torch.Tensor, shape = (128, n_frames)
            包含Mel频谱的Tensor
        """
        # 计算输入音频的log-mel谱特征
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs()**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
