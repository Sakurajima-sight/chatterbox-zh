from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor

from .perceiver import Perceiver
from .t3_config import T3Config


@dataclass
class T3Cond:
    """
    该数据类用于保存所有或大部分的条件信息，如说话人嵌入、CLAP（拍手）、情感等。
    TODO：序列化方法目前没有使用，保留它们是为了方便使用。
    """

    speaker_emb: Tensor  # 说话人嵌入（Tensor类型）
    clap_emb: Optional[Tensor] = None  # CLAP嵌入（可选）
    cond_prompt_speech_tokens: Optional[Tensor] = None  # 条件提示语音的token（可选）
    cond_prompt_speech_emb: Optional[Tensor] = None  # 条件提示语音的嵌入（可选）
    emotion_adv: Optional[Tensor] = 0.5  # 情感调整因子（默认为0.5）

    def to(self, *, device=None, dtype=None):
        """
        将该对象中的所有Tensor转移到指定的设备（如GPU）并转换为指定的数据类型（dtype）。

        :param device: 目标设备
        :param dtype: 目标数据类型
        :return: 转换后的T3Cond对象
        """
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = type(v.view(-1)[0].item()) is not int  # 判断是否是浮点数
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self

    def save(self, fpath):
        """
        将当前对象保存到指定路径。

        :param fpath: 保存路径
        """
        torch.save(self.__dict__, fpath)

    @staticmethod
    def load(fpath, map_location="cpu"):
        """
        从指定路径加载T3Cond对象。

        :param fpath: 加载路径
        :param map_location: 指定设备（如cpu或gpu）
        :return: 加载的T3Cond对象
        """
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return T3Cond(**kwargs)


class T3CondEnc(nn.Module):
    """
    处理所有非文本的条件信息，例如说话人嵌入、提示、CLAP、情感等。
    """

    def __init__(self, hp: T3Config):
        """
        初始化T3CondEnc模块。

        :param hp: 配置对象，包含所有超参数
        """
        super().__init__()
        self.hp = hp

        # 根据配置选择不同的编码器类型
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)  # 说话人嵌入投影
        else:
            raise NotImplementedError(str(hp.encoder_type))

        # 如果使用情感调整
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # 如果使用Perceiver Resampler
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()

    def forward(self, cond: T3Cond):
        """
        前向传播，处理所有条件信息并返回合并后的条件嵌入。

        :param cond: T3Cond对象，包含所有条件信息
        :return: 合并后的条件嵌入
        """

        # 验证条件中的提示和嵌入是否一致
        assert (cond.cond_prompt_speech_tokens is None) == (cond.cond_prompt_speech_emb is None), \
            "no embeddings for cond_prompt_speech_tokens"

        # 说话人嵌入投影
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]  # (B, 1, dim)
        empty = torch.zeros_like(cond_spkr[:, :0])  # (B, 0, dim) 空的张量

        # TODO CLAP: 暂时未实现CLAP的处理
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)

        # 条件提示嵌入
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # 情感调整：如果模型使用情感条件，必须提供情感调整信息
        cond_emotion_adv = empty  # (B, 0, dim)
        if self.hp.emotion_adv:
            assert cond.emotion_adv is not None
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))

        # 将所有条件嵌入连接在一起并返回
        cond_embeds = torch.cat((
            cond_spkr,  # 说话人嵌入
            cond_clap,  # CLAP嵌入
            cond_prompt_speech_emb,  # 条件提示嵌入
            cond_emotion_adv,  # 情感调整嵌入
        ), dim=1)
        return cond_embeds
