# Adapted from https://github.com/CorentinJ/Real-Time-Voice-Cloning
# MIT License
from typing import List, Union, Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided
import librosa
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .config import VoiceEncConfig
from .melspec import melspectrogram


def pack(arrays, seq_len: int = None, pad_value=0):
    """
    将长度为B的多个数组对象按维度合并成一个张量，形状为(B, T, ...)。如果数组长度不一致，按最大长度进行填充。

    参数:
        arrays: 包含多个数组对象的列表，这些数组的形状除了第一个轴外，其余轴应该一致。
        seq_len: T的值，必须大于等于所有数组长度的最大值。如果为None，则默认为最大长度。
        pad_value: 填充值，默认为0，用于填充数组中缺失的部分。

    返回:
        返回形状为(B, T, ...)的张量。
    """
    if seq_len is None:
        seq_len = max(len(array) for array in arrays)
    else:
        assert seq_len >= max(len(array) for array in arrays)

    # 将列表转换为np.array
    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]

    # 转换为张量，并处理设备（如GPU或CPU）
    device = None
    if isinstance(arrays[0], torch.Tensor):
        tensors = arrays
        device = tensors[0].device
    else:
        tensors = [torch.as_tensor(array) for array in arrays]

    # 填充张量并返回
    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(packed_shape, pad_value, dtype=tensors[0].dtype, device=device)

    for i, tensor in enumerate(tensors):
        packed_tensor[i, :tensor.size(0)] = tensor

    return packed_tensor


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    """
    计算语音帧的窗口数和目标长度。

    参数:
        n_frames: 输入的帧数。
        step: 每个窗口的步长。
        min_coverage: 每个窗口的最小覆盖比例。
        hp: VoiceEncConfig对象，包含相关的配置信息。

    返回:
        n_wins: 窗口数。
        target_n: 目标帧数。
    """
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(
    overlap: float,
    rate: float,
    hp: VoiceEncConfig,
):
    """
    计算两个部分性语音片段之间的帧步长。

    参数:
        overlap: 重叠比例，范围为[0, 1)。
        rate: 采样率或者语速。
        hp: VoiceEncConfig对象，包含相关的配置信息。

    返回:
        frame_step: 计算得到的帧步长。
    """
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


def stride_as_partials(
    mel: np.ndarray,
    hp: VoiceEncConfig,
    overlap=0.5,
    rate: float = None,
    min_coverage=0.8,
):
    """
    将未缩放的梅尔频谱图切分成多个部分性（partial）梅尔频谱图，按帧步长重叠。
    
    参数:
        mel: 输入的梅尔频谱图，形状为(T, M)，其中T是时间帧数，M是梅尔频率数。
        hp: VoiceEncConfig对象，包含相关的配置信息。
        overlap: 帧间重叠比例，默认为0.5。
        rate: 采样速率（可选），默认情况下为None。
        min_coverage: 每个部分的最小覆盖率，默认为0.8。
        
    返回:
        partials: 切分后的多个部分梅尔频谱图，形状为(N, P, M)，N是部分数量，P是每部分的帧数，M是梅尔频率数。
    """
    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)

    # 计算梅尔频谱图中可以容纳多少个部分
    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    # 修剪或填充梅尔频谱图，以匹配部分的数量
    if target_len > len(mel):
        mel = np.concatenate((mel, np.full((target_len - len(mel), hp.num_mels), 0)))
    elif target_len < len(mel):
        mel = mel[:target_len]

    # 确保数据是float32并且在内存中是连续的
    mel = mel.astype(np.float32, order="C")

    # 使用strides将数组重新排列成(N, P, M)形状，其中N是部分数量，P是每部分的帧数，M是梅尔频率数
    shape = (n_partials, hp.ve_partial_frames, hp.num_mels)
    strides = (mel.strides[0] * frame_step, mel.strides[0], mel.strides[1])
    partials = as_strided(mel, shape, strides)
    return partials


class VoiceEncoder(nn.Module):
    """
    语音编码器，用于生成语音的嵌入向量（Speaker Embedding）。可以通过Mel频谱图或音频波形生成说话人的语音特征。
    """

    def __init__(self, hp=VoiceEncConfig()):
        """
        初始化语音编码器。

        :param hp: 配置对象，包含所有超参数（如LSTM的大小、嵌入向量的大小等）。
        """
        super().__init__()

        self.hp = hp

        # 网络定义
        self.lstm = nn.LSTM(self.hp.num_mels, self.hp.ve_hidden_size, num_layers=3, batch_first=True)
        if hp.flatten_lstm_params:
            self.lstm.flatten_parameters()
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        # 余弦相似度缩放（固定初始参数值）
        self.similarity_weight = nn.Parameter(torch.tensor([10.]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]), requires_grad=True)

    @property
    def device(self):
        """
        获取模型当前所在的设备（GPU或CPU）。
        """
        return next(self.parameters()).device

    def forward(self, mels: torch.FloatTensor):
        """
        计算一批部分话语的嵌入向量。

        :param mels: 一个批次的未经缩放的Mel频谱图，形状为(B, T, M)，其中T是帧数，M是Mel频率数量
        :return: 计算得到的嵌入向量，形状为(B, E)，E为speaker_embed_size。嵌入向量经过L2归一化，值范围在[-1, 1]之间。
        """
        if self.hp.normalized_mels and (mels.min() < 0 or mels.max() > 1):
            raise Exception(f"Mels outside [0, 1]. Min={mels.min()}, Max={mels.max()}")

        # 通过LSTM层进行前向传播
        _, (hidden, _) = self.lstm(mels)

        # 投影到最终的隐藏状态
        raw_embeds = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)

        # L2归一化嵌入向量
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def inference(self, mels: torch.Tensor, mel_lens, overlap=0.5, rate: float=None, min_coverage=0.8, batch_size=None):
        """
        计算一批完整话语的嵌入向量，并返回带有梯度的结果。

        :param mels: (B, T, M) 形状的未经缩放的Mel频谱图
        :return: (B, E) 形状的嵌入向量（在CPU上）
        """
        mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens

        # 计算如何将话语分割成多个部分
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials, target_lens = zip(*(get_num_wins(l, frame_step, min_coverage, self.hp) for l in mel_lens))

        # 可能需要对mels进行填充
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.full((mels.size(0), len_diff, self.hp.num_mels), 0, dtype=torch.float32)
            mels = torch.cat((mels, pad.to(mels.device)), dim=1)

        # 将所有部分合并，方便批量处理
        partials = [
            mel[i * frame_step: i * frame_step + self.hp.ve_partial_frames]
            for mel, n_partial in zip(mels, n_partials) for i in range(n_partial)
        ]
        assert all(partials[0].shape == partial.shape for partial in partials)
        partials = torch.stack(partials)

        # 前向传播计算所有部分
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        partial_embeds = torch.cat([self(batch) for batch in partials.chunk(n_chunks)], dim=0).cpu()

        # 将部分嵌入向量聚合成完整嵌入，并进行L2归一化
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = [torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:])]
        raw_embeds = torch.stack(raw_embeds)
        embeds = raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

        return embeds

    @staticmethod
    def utt_to_spk_embed(utt_embeds: np.ndarray):
        """
        从L2归一化的语音嵌入中计算说话人嵌入。

        :param utt_embeds: (N, E) 形状的语音嵌入数组
        :return: L2归一化的说话人嵌入，形状为(E,)
        """
        assert utt_embeds.ndim == 2
        utt_embeds = np.mean(utt_embeds, axis=0)
        return utt_embeds / np.linalg.norm(utt_embeds, 2)

    @staticmethod
    def voice_similarity(embeds_x: np.ndarray, embeds_y: np.ndarray):
        """
        计算两组L2归一化的语音嵌入向量之间的余弦相似度。

        :param embeds_x: 语音嵌入向量x
        :param embeds_y: 语音嵌入向量y
        :return: 余弦相似度值
        """
        embeds_x = embeds_x if embeds_x.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_x)
        embeds_y = embeds_y if embeds_y.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_y)
        return embeds_x @ embeds_y

    def embeds_from_mels(
        self, mels: Union[Tensor, List[np.ndarray]], mel_lens=None, as_spk=False, batch_size=32, **kwargs
    ):
        """
        从Mel频谱图计算语音嵌入或说话人嵌入。

        :param mels: 一个批次的Mel频谱图数据，或者包含多个(长度, Mel数量)数组的列表。
        :param mel_lens: 传递Mel数据时的每个Mel的长度
        :param as_spk: 是否返回说话人嵌入（默认为False，返回语音嵌入）
        :param kwargs: 传递给inference()的其他参数
        :returns: 语音或说话人嵌入，形状为(B, E)或(E,)
        """
        # 加载并打包Mel频谱图
        if isinstance(mels, List):
            mels = [np.asarray(mel) for mel in mels]
            assert all(m.shape[1] == mels[0].shape[1] for m in mels), "Mels aren't in (B, T, M) format"
            mel_lens = [mel.shape[0] for mel in mels]
            mels = pack(mels)

        # 获取嵌入
        with torch.inference_mode():
            utt_embeds = self.inference(mels.to(self.device), mel_lens, batch_size=batch_size, **kwargs).numpy()

        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wavs(
        self,
        wavs: List[np.ndarray],
        sample_rate,
        as_spk=False,
        batch_size=32,
        trim_top_db: Optional[float]=20,
        **kwargs
    ):
        """
        包装器，用于从音频波形生成语音嵌入。

        :param wavs: 音频波形列表
        :param sample_rate: 原始音频的采样率
        :param trim_top_db: 可选，音频预处理时的顶部去噪参数
        :param kwargs: 传递给embeds_from_mels的其他参数
        """
        if sample_rate != self.hp.sample_rate:
            wavs = [
                librosa.resample(wav, orig_sr=sample_rate, target_sr=self.hp.sample_rate, res_type="kaiser_fast")
                for wav in wavs
            ]

        if trim_top_db:
            wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]

        if "rate" not in kwargs:
            kwargs["rate"] = 1.3  # Resemble的默认值

        mels = [melspectrogram(w, self.hp).T for w in wavs]

        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, **kwargs)
