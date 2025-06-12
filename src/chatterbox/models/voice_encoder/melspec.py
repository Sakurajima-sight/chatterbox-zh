from functools import lru_cache

from scipy import signal
import numpy as np
import librosa


@lru_cache()
def mel_basis(hp):
    """
    生成Mel频率滤波器组的基矩阵。
    
    参数:
        hp: 包含音频处理参数的对象，包括采样率、FFT大小、梅尔频率数等。
        
    返回:
        mel_basis: 一个二维矩阵，形状为 (num_mels, num_freq)，用于将频谱转换为Mel频谱。
    """
    assert hp.fmax <= hp.sample_rate // 2  # 最大频率不应超过采样率的一半
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax)  # -> (nmel, nfreq)


def preemphasis(wav, hp):
    """
    对音频信号进行预加重处理。
    
    参数:
        wav: 输入的音频信号，通常为一个一维的numpy数组。
        hp: 包含音频处理参数的对象，包含预加重系数等。
        
    返回:
        预加重后的音频信号。
    """
    assert hp.preemphasis != 0  # 预加重系数不应为0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)  # 使用滤波器进行预加重
    wav = np.clip(wav, -1, 1)  # 确保音频信号在[-1, 1]范围内
    return wav


def melspectrogram(wav, hp, pad=True):
    """
    计算梅尔频谱图。
    
    参数:
        wav: 输入的音频信号。
        hp: 包含音频处理参数的对象，包含预加重系数、FFT大小、梅尔频率数等。
        pad: 是否对STFT进行填充。
        
    返回:
        mel: 计算得到的梅尔频谱图，形状为(M, T)，M是梅尔频率数，T是时间帧数。
    """
    # 如果需要预加重，先进行预加重处理
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07  # 确保音频信号的范围正确

    # 计算短时傅里叶变换（STFT）
    spec_complex = _stft(wav, hp, pad=pad)

    # 计算频谱的幅度
    spec_magnitudes = np.abs(spec_complex)

    # 如果设置了mel_power, 对幅度进行幂运算
    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # 计算梅尔频谱并转换为dB（如果需要）
    mel = np.dot(mel_basis(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    # 如果需要，将梅尔频谱归一化
    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(np.float32)

    # Sanity check: 确保mel的形状正确
    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size
    return mel   # (M, T)


def _stft(y, hp, pad=True):
    """
    计算短时傅里叶变换（STFT）。
    
    参数:
        y: 输入的音频信号。
        hp: 包含音频处理参数的对象，包含FFT大小、窗口大小、hop_size等。
        pad: 是否对信号进行填充。
        
    返回:
        spec_complex: 复数频谱。
    """
    return librosa.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",  # 采用“反射”填充模式
    )


def _amp_to_db(x, hp):
    """
    将幅度谱转换为分贝（dB）谱。
    
    参数:
        x: 幅度谱。
        hp: 包含音频处理参数的对象，包含最小STFT幅度。
        
    返回:
        dB谱：幅度谱转换后的分贝值。
    """
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _db_to_amp(x):
    """
    将分贝（dB）谱转换回幅度谱。
    
    参数:
        x: 分贝谱。
        
    返回:
        幅度谱。
    """
    return np.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    """
    将梅尔频谱从dB值归一化到0到1的范围。
    
    参数:
        s: 输入的梅尔频谱。
        hp: 包含音频处理参数的对象，包含最小STFT幅度。
        headroom_db: 设置归一化后的频谱的头部空间。
        
    返回:
        归一化后的梅尔频谱。
    """
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)  # 计算最小幅度的dB值
    s = (s - min_level_db) / (-min_level_db + headroom_db)  # 进行归一化
    return s
