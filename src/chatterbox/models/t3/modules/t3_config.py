from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    """
    配置类T3Config，定义了用于T3模型的各种超参数和配置选项。它包括文本和语音的token、模型的配置以及一些条件编码器的参数。
    """

    start_text_token = 0  # 文本开始token的ID
    stop_text_token = 255  # 文本结束token的ID
    text_tokens_dict_size = 3049  # 文本token字典的大小
    max_text_tokens = 2048  # 最大文本token数量

    start_speech_token = 6561  # 语音开始token的ID
    stop_speech_token = 6562  # 语音结束token的ID
    speech_tokens_dict_size = 8194  # 语音token字典的大小
    max_speech_tokens = 4096  # 最大语音token数量

    llama_config_name = "Llama_520M"  # 使用的Llama模型配置名称
    input_pos_emb = "learned"  # 输入位置嵌入类型（学习得到的）
    speech_cond_prompt_len = 150  # 语音条件提示的长度

    # 以下是针对T3CondEnc的配置
    encoder_type = "voice_encoder"  # 编码器类型，使用语音编码器
    speaker_embed_size = 256  # 说话人嵌入的维度
    use_perceiver_resampler = True  # 是否使用Perceiver重采样器
    emotion_adv = True  # 是否启用情感调节

    @property
    def n_channels(self):
        """
        返回模型的隐藏层维度（来自LLAMA配置）。
        """
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
