from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

def punc_norm(text: str) -> str:
    """
    用于清理和规范化文本中的标点符号。这个函数可以清理来自大语言模型的文本，
    或包含不常见字符的数据集中的文本，主要用于标点符号的清理和替换。

    参数:
    text (str): 输入的文本字符串
    
    返回:
    str: 规范化后的文本
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."  # 如果输入为空，返回提示语句

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]  # 将首字母大写

    # Remove multiple space chars
    text = " ".join(text.split())  # 移除多余的空格

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),      # 替换省略号
        ("…", ", "),        # 替换中文省略号
        (":", ","),         # 替换冒号为逗号
        (" - ", ", "),      # 替换破折号为空格逗号
        (";", ", "),        # 替换分号为逗号
        ("—", "-"),         # 替换长破折号为短破折号
        ("–", "-"),         # 替换短破折号为短破折号
        (" ,", ","),        # 替换多余空格和逗号
        ("“", "\""),        # 替换中文双引号为英文双引号
        ("”", "\""),        # 替换中文双引号为英文双引号
        ("‘", "'"),         # 替换中文单引号为英文单引号
        ("’", "'"),         # 替换中文单引号为英文单引号
        ("《", ","),        # 替换中文书名号左为逗号
        ("》", ","),        # 替换中文书名号右为逗号
    ]

    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)  # 执行替换

    # Add full stop if no ending punc
    text = text.rstrip(" ")  # 去掉文本末尾的空格
    sentence_enders = {".", "!", "?", "-", ","}  # 定义结束标点符号
    if not any(text.endswith(p) for p in sentence_enders):  # 如果文本没有以标点符号结尾
        text += "."  # 添加句号

    return text


@dataclass
class Conditionals:
    """
    T3 和 S3Gen 的条件数据类。
    - T3 条件：
        - speaker_emb: 说话人嵌入
        - clap_emb: 鼓掌嵌入
        - cond_prompt_speech_tokens: 条件语音文本的 token
        - cond_prompt_speech_emb: 条件语音文本的嵌入
        - emotion_adv: 情感增强
    - S3Gen 条件：
        - prompt_token: 提示 token
        - prompt_token_len: 提示 token 长度
        - prompt_feat: 提示特征
        - prompt_feat_len: 提示特征长度
        - embedding: 嵌入
    """
    t3: T3Cond  # T3 模型的条件
    gen: dict  # S3Gen 模型的条件，存储为字典

    def to(self, device):
        """
        将条件数据转移到指定设备（如 GPU 或 CPU）。
        
        参数:
        device (torch.device): 目标设备
        
        返回:
        Conditionals: 返回转移到指定设备后的条件数据对象
        """
        self.t3 = self.t3.to(device=device)  # 将 T3 条件数据转移到设备
        for k, v in self.gen.items():  # 遍历 gen 字典中的每个条件
            if torch.is_tensor(v):  # 如果条件是张量
                self.gen[k] = v.to(device=device)  # 将该条件转移到指定设备
        return self

    def save(self, fpath: Path):
        """
        将条件数据保存到文件中。
        
        参数:
        fpath (Path): 保存文件的路径
        """
        arg_dict = dict(
            t3=self.t3.__dict__,  # 将 t3 条件数据保存为字典
            gen=self.gen  # 保存 gen 条件数据
        )
        torch.save(arg_dict, fpath)  # 将字典保存为文件

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        """
        从文件中加载条件数据。
        
        参数:
        fpath (Path): 条件数据文件路径
        map_location (str or torch.device): 将数据加载到的设备，默认为 "cpu"
        
        返回:
        Conditionals: 加载后的条件数据对象
        """
        if isinstance(map_location, str):
            map_location = torch.device(map_location)  # 将字符串转换为设备对象
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)  # 加载数据
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])  # 使用加载的数据创建并返回 Conditionals 实例


class ChatterboxTTS:
    """
    ChatterboxTTS 类用于文本到语音（TTS）的生成，结合了 T3 和 S3Gen 模型，
    以及语音编码器和标记器，通过条件数据生成与输入文本相关的语音。
    """
    ENC_COND_LEN = 6 * S3_SR  # 编码器条件长度
    DEC_COND_LEN = 10 * S3GEN_SR  # 解码器条件长度

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        """
        初始化 ChatterboxTTS 类。

        参数:
        t3 (T3): T3 模型
        s3gen (S3Gen): S3Gen 模型
        ve (VoiceEncoder): 语音编码器
        tokenizer (EnTokenizer): 标记器
        device (str): 设备（如 'cpu' 或 'cuda'）
        conds (Conditionals, optional): 条件数据，默认值为 None
        """
        self.sr = S3GEN_SR  # 生成的音频采样率
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()  # 水印处理器

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        """
        从本地检查点文件加载 ChatterboxTTS 模型。

        参数:
        ckpt_dir (str): 检查点文件所在的目录路径
        device (str): 设备（如 'cpu' 或 'cuda'）

        返回:
        ChatterboxTTS: 加载的 ChatterboxTTS 实例
        """
        ckpt_dir = Path(ckpt_dir)

        # 首先加载到 CPU，以处理 CUDA 保存的模型
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        # 加载语音编码器模型
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        # 加载 T3 模型
        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        # 加载 S3Gen 模型
        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        # 加载标记器
        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        # 加载条件数据
        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        """
        从 Hugging Face 上加载预训练模型。

        参数:
        device (str): 设备（如 'cpu' 或 'cuda'）

        返回:
        ChatterboxTTS: 加载的 ChatterboxTTS 实例
        """
        # 检查 macOS 是否支持 MPS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS 不可用，因为当前的 PyTorch 安装没有启用 MPS。")
            else:
                print("MPS 不可用，因为当前的 macOS 版本不满足要求，或者设备不支持 MPS。")
            device = "cpu"

        # 从 Hugging Face 下载模型文件
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """
        准备生成语音所需的条件数据。

        参数:
        wav_fpath (str): 参考语音文件的路径
        exaggeration (float): 情感增强的倍数，默认值为 0.5
        """
        # 加载参考音频
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # 语音条件的提示 tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # 语音编码器的说话人嵌入
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        # 准备 T3 条件数据
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        """
        生成语音从文本或语音提示。

        参数:
        text (str): 输入文本
        audio_prompt_path (str, optional): 参考音频文件路径
        exaggeration (float): 情感增强倍数，默认值为 0.5
        cfg_weight (float): 配置权重，默认值为 0.5
        temperature (float): 温度值，默认值为 0.8

        返回:
        torch.Tensor: 生成的水印语音
        """
        # 如果有音频提示，准备条件数据
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "请先执行 `prepare_conditionals` 或指定 `audio_prompt_path`"

        # 更新情感增强值
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # 文本预处理和标记化
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        # 配置权重设置
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # 需要两个序列用于 CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # 使用 T3 模型进行推理
        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: 使用配置中的值
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # 提取条件批次
            speech_tokens = speech_tokens[0]

            # 去除无效的 tokens
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            # 使用 S3Gen 模型生成音频
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
