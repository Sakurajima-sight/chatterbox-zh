"""mel-spectrogram extraction in Matcha-TTS
该代码实现了通过STFT和Mel滤波器生成Mel频谱图的功能。通常用于语音处理任务，特别是语音合成。
"""
from librosa.filters import mel as librosa_mel_fn
import torch
import numpy as np


# NOTE: 这些是全局变量
mel_basis = {}
hann_window = {}

# 动态范围压缩，用于log压缩振幅
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    对输入信号进行动态范围压缩。常用于声音信号处理中，避免小值造成的影响。
    
    参数：
        x: 输入的张量
        C: 放大因子
        clip_val: 最小值用于截断
    返回：
        压缩后的张量
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


# 频谱归一化函数
def spectral_normalize_torch(magnitudes):
    """
    对幅度谱进行归一化（即动态范围压缩）
    
    参数：
        magnitudes: 输入的幅度谱
    返回：
        归一化后的幅度谱
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output

"""
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: 24000
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False

这是一个配置样本，定义了用于生成Mel频谱图的超参数。
"""

def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
    """
    该函数用于从音频波形y中提取Mel频谱图，适用于语音合成等任务。
    
    参数：
        y: 输入的音频信号（1D数组或2D数组）
        n_fft: FFT大小（默认1920）
        num_mels: Mel滤波器数量（默认80）
        sampling_rate: 采样率（默认24000）
        hop_size: hop大小（默认480）
        win_size: 窗口大小（默认1920）
        fmin: Mel滤波器的最低频率（默认0）
        fmax: Mel滤波器的最高频率（默认8000）
        center: 是否中心对齐（默认False）

    返回：
        spec: 提取的Mel频谱图（Torch张量）
    """

    # 如果y是numpy数组，则转换为Torch张量
    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()

    # 如果y是一维的，扩展为二维张量
    if len(y.shape) == 1:
        y = y[None, ]

    # 用于调试：检查输入信号的最小最大值
    if torch.min(y) < -1.0:
        pass#print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        pass#print("max value is ", torch.max(y))

    # 使用全局变量初始化mel_basis和hann_window（避免每次都重新计算）
    global mel_basis, hann_window  # pylint: disable=global-statement,global-variable-not-assigned
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        # 计算Mel滤波器
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # 对音频信号进行填充，确保能够进行STFT操作
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    # 进行STFT操作，返回复数频谱
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    # 计算幅度谱
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    # 将Mel滤波器应用于频谱
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)

    # 对Mel频谱进行归一化处理
    spec = spectral_normalize_torch(spec)

    return spec
