class VoiceEncConfig:
    """
    语音编码器配置类
    该类包含用于语音编码器的各种超参数配置，包括梅尔频谱、LSTM结构、STFT参数等。
    """
    num_mels = 40  # 梅尔频谱的频带数量，常见值为 40-80，决定了梅尔频谱的维度
    sample_rate = 16000  # 音频采样率，通常是 16000 或 22050，影响音频的时间分辨率
    speaker_embed_size = 256  # 说话人嵌入向量的维度，用于表示不同说话人的特征
    ve_hidden_size = 256  # 语音编码器的隐藏层大小，影响模型的容量和性能
    flatten_lstm_params = False  # 是否展平LSTM的参数（不常用，一般情况下保留为False）
    n_fft = 400  # FFT大小，影响STFT计算中频谱分析的频率分辨率
    hop_size = 160  # 每次STFT计算之间的步长，影响时间分辨率
    win_size = 400  # 每帧的窗口大小，影响STFT计算中的时域窗口
    fmax = 8000  # 梅尔频谱的最大频率，通常是采样率的一半
    fmin = 0  # 梅尔频谱的最小频率，通常从0Hz开始
    preemphasis = 0.  # 预加重系数，通常用于提升高频部分的能量
    mel_power = 2.0  # 梅尔频谱的幂次，通常为2.0，影响梅尔频谱的能量分布
    mel_type = "amp"  # 梅尔频谱的类型，'amp'表示幅度谱，'log'表示对数谱
    normalized_mels = False  # 是否对梅尔频谱进行归一化处理，影响梅尔频谱的数值范围
    ve_partial_frames = 160  # 每次输入给语音编码器的帧数，影响每次处理的时长
    ve_final_relu = True  # 是否在语音编码器的最终输出使用ReLU激活函数
    stft_magnitude_min = 1e-4  # STFT幅度的最小值，用于避免计算中的零幅度
