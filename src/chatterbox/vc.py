from pathlib import Path

import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    """
    ChatterboxVC 类用于语音转换（Voice Conversion），
    将输入的音频转换为目标说话人的语音，基于 S3Gen 模型进行处理。
    该类支持从本地或预训练模型加载，生成带水印的语音输出。
    """
    ENC_COND_LEN = 6 * S3_SR  # 编码器条件长度
    DEC_COND_LEN = 10 * S3GEN_SR  # 解码器条件长度

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict = None,
    ):
        """
        初始化 ChatterboxVC 类。

        参数:
        s3gen (S3Gen): S3Gen 模型，用于语音转换的核心模型
        device (str): 设备（如 'cpu' 或 'cuda'）
        ref_dict (dict, optional): 参考条件数据，默认值为 None
        """
        self.sr = S3GEN_SR  # 生成音频的采样率
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()  # 水印处理器
        self.ref_dict = ref_dict.to(device) if ref_dict else None  # 参考条件数据，若无则为 None

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        """
        从本地检查点文件加载 ChatterboxVC 模型。

        参数:
        ckpt_dir (str): 检查点文件所在的目录路径
        device (str): 设备（如 'cpu' 或 'cuda'）

        返回:
        ChatterboxVC: 加载的 ChatterboxVC 实例
        """
        ckpt_dir = Path(ckpt_dir)
        
        # 加载到 CPU 以处理 CUDA 保存的模型
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        # 如果条件文件存在，加载其中的参考条件数据
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        # 加载 S3Gen 模型
        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()  # 将模型转移到指定设备并设置为评估模式

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        """
        从 Hugging Face 上加载预训练的模型。

        参数:
        device (str): 设备（如 'cpu' 或 'cuda'）

        返回:
        ChatterboxVC: 加载的 ChatterboxVC 实例
        """
        # 检查 macOS 是否支持 MPS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS 不可用，因为当前的 PyTorch 安装没有启用 MPS。")
            else:
                print("MPS 不可用，因为当前的 macOS 版本不满足要求，或者设备不支持 MPS。")
            device = "cpu"
        
        # 从 Hugging Face 下载模型文件
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpath):
        """
        设置目标语音，即将参考音频加载并提取其特征。

        参数:
        wav_fpath (str): 参考音频文件的路径
        """
        # 加载参考音频文件
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]  # 裁剪音频长度
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        """
        生成目标语音，将输入音频转换为目标说话人的语音。

        参数:
        audio (str): 输入音频文件路径
        target_voice_path (str, optional): 目标说话人的参考音频路径

        返回:
        torch.Tensor: 转换后的目标语音
        """
        # 如果指定了目标音频，设置目标声纹特征
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "请先执行 `set_target_voice` 或指定 `target_voice_path`"

        with torch.inference_mode():
            # 加载输入音频并转换为 16k 采样率
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            # 获取输入音频的 S3Gen tokens
            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            # 使用 S3Gen 模型生成目标语音
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            # 应用水印处理
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        return torch.from_numpy(watermarked_wav).unsqueeze(0)
