import argparse
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)
from transformers import TrainingArguments as HfTrainingArguments
from datasets import load_dataset, DatasetDict, VerificationMode, Audio
import datasets

# Chatterbox imports
from chatterbox.tts import ChatterboxTTS # To load the full model initially
from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3gen.s3gen import S3Token2Mel, mel_spectrogram # S3GEN_SR for mels
from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR as S3_TOKENIZER_SR # S3Tokenizer operates at 16kHz
from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec # The actual module we want to finetune
from chatterbox.models.s3gen.xvector import CAMPPlus # Speaker encoder used by S3Gen

logger = logging.getLogger(__name__)

# --- Training Arguments (can reuse CustomTrainingArguments) ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    """
    自定义训练参数类，继承自HuggingFace的TrainingArguments，用于扩展早停（early stopping）等功能。
    """
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Enable early stopping."}
    )

# --- Model Arguments ---
@dataclass
class S3GenModelArguments:
    """
    S3Gen模型相关参数定义，包含模型路径、本地目录及各模块是否冻结等设置。
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to base Chatterbox model"})
    local_model_dir: Optional[str] = field(default=None, metadata={"help": "Path to local Chatterbox model directory"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache directory"})
    # S3Gen specific finetuning args
    freeze_speaker_encoder: bool = field(default=True, metadata={"help": "Freeze S3Gen's internal speaker encoder (CAMPPlus)."})
    freeze_s3_tokenizer: bool = field(default=True, metadata={"help": "Freeze S3Gen's internal S3Tokenizer."})
    # The 'flow' part of S3Gen will be trained. HiFiGAN part will be frozen.

# --- Data Arguments ---
@dataclass
class S3GenDataArguments:
    """
    数据加载和预处理相关参数定义。包含数据集名称、分割、音频字段名等，适配S3Gen训练所需的数据结构。
    """
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF Dataset name"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "HF Dataset config name"})
    train_split_name: str = field(default="train", metadata={"help": "Train split name"})
    eval_split_name: Optional[str] = field(default="validation", metadata={"help": "Eval split name"})
    audio_column_name: str = field(default="audio", metadata={"help": "Audio column in dataset"})
    # No text column needed for S3Gen training directly, but might be in dataset
    
    max_speech_token_len: int = field(default=750, metadata={"help": "Max S3 speech tokens for target sequence."})
    max_mel_len: int = field(default=1500, metadata={"help": "Max mel frames for target mel (max_speech_token_len * 2)"})
    # For CausalMaskedDiffWithXvec, we need prompt_tokens and prompt_feats for conditioning
    prompt_audio_duration_s: float = field(default=3.0, metadata={"help": "Duration of audio for prompt_token and prompt_feat."})
    
    eval_split_size: float = field(default=0.01, metadata={"help": "Eval split fraction"}) # Increased default
    ignore_verifications: bool = field(default=False, metadata={"help":"Ignore dataset verifications."})


# --- S3Gen Finetuning Dataset ---
class S3GenFineTuningDataset(Dataset):
    """
    S3Gen微调用的数据集封装类。
    用于从HuggingFace数据集抽取、处理和转换为S3Gen微调所需的各种输入特征（如梅尔谱、S3语音token、说话人嵌入等）。
    """
    def __init__(
        self,
        data_args: S3GenDataArguments,
        s3gen_model: S3Gen, # 直接传入S3Gen模型，包括tokenizer、mel_extractor、speaker_encoder
        hf_dataset: datasets.Dataset
    ):
        """
        初始化数据集，保存配置参数、模型、数据集，准备采样器。
        """
        self.data_args = data_args
        self.s3gen_model = s3gen_model # 包含tokenizer、mel_extractor、speaker_encoder
        self.dataset_source = hf_dataset

        self.s3_tokenizer_sr = S3_TOKENIZER_SR # S3Tokenizer的采样率（16kHz）
        self.s3_gen_native_sr = S3GEN_SR     # S3Gen梅尔谱、CAMPPlus输入采样率（24kHz）
        
        self.prompt_audio_samples_16k = int(data_args.prompt_audio_duration_s * self.s3_tokenizer_sr)
        self.prompt_audio_samples_24k = int(data_args.prompt_audio_duration_s * self.s3_gen_native_sr)

        self._resamplers = {}

    def _get_resampler(self, orig_sr: int, target_sr: int) -> T.Resample:
        """
        获取或创建采样率转换器（Resample），实现不同采样率的音频重采样。
        """
        if (orig_sr, target_sr) not in self._resamplers:
            self._resamplers[(orig_sr, target_sr)] = T.Resample(orig_sr, target_sr)
        return self._resamplers[(orig_sr, target_sr)]

    def __len__(self):
        """
        返回数据集样本总数。
        """
        return len(self.dataset_source)

    def _load_and_preprocess_audio(self, audio_data_from_hf):
        """
        加载并预处理音频数据，转换为16kHz和24kHz的Tensor，适配S3Tokenizer与梅尔谱提取器的输入需求。
        返回：(wav_16k_tensor, wav_24k_tensor)
        """
        waveform: Optional[torch.Tensor] = None
        original_sr: Optional[int] = None

        if isinstance(audio_data_from_hf, str):
            try: waveform, original_sr = torchaudio.load(audio_data_from_hf)
            except Exception: return None, None
        elif isinstance(audio_data_from_hf, dict) and "array" in audio_data_from_hf and "sampling_rate" in audio_data_from_hf:
            np_array = audio_data_from_hf["array"]
            if not isinstance(np_array, np.ndarray): return None, None
            waveform = torch.from_numpy(np_array).float()
            original_sr = audio_data_from_hf["sampling_rate"]
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: return None, None

        if waveform is None or original_sr is None or waveform.numel() == 0: return None, None
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 保证float32类型
        waveform = waveform.float()

        # 16kHz音频（供S3Tokenizer和部分说话人编码器用）
        if original_sr != self.s3_tokenizer_sr:
            resampler_16k = self._get_resampler(original_sr, self.s3_tokenizer_sr)
            wav_16k_tensor = resampler_16k(waveform)
        else:
            wav_16k_tensor = waveform.clone() # 克隆，避免修改原数据
        
        # 24kHz音频（供梅尔谱提取器和CAMPPlus用）
        if original_sr != self.s3_gen_native_sr:
            resampler_24k = self._get_resampler(original_sr, self.s3_gen_native_sr)
            wav_24k_tensor = resampler_24k(waveform)
        else:
            wav_24k_tensor = waveform.clone()

        return wav_16k_tensor.squeeze(0), wav_24k_tensor.squeeze(0) # 返回长度为L的Tensor

    def __getitem__(self, idx) -> Optional[Dict[str, torch.Tensor]]:
        """
        取出单条数据并全部转换为模型所需输入，包括音频特征、tokens、说话人向量、提示特征等。
        失败时返回None（例如音频异常）。
        """
        item = self.dataset_source[idx]
        audio_data_hf = item[self.data_args.audio_column_name]

        wav_16k_tensor, wav_24k_tensor = self._load_and_preprocess_audio(audio_data_hf)
        if wav_16k_tensor is None or wav_24k_tensor is None or wav_16k_tensor.numel() == 0 or wav_24k_tensor.numel() == 0:
            return None

        # 1. 计算目标梅尔谱（speech_feat），用24kHz音频
        try:
            target_mel = self.s3gen_model.mel_extractor(wav_24k_tensor.unsqueeze(0)).squeeze(0).transpose(0, 1) # (T_mel, F)
            if target_mel.size(0) > self.data_args.max_mel_len:
                target_mel = target_mel[:self.data_args.max_mel_len, :]
            speech_feat_len = torch.tensor(target_mel.size(0), dtype=torch.long)
        except Exception as e:
            logger.error(f"Item {idx}: Error extracting target_mel: {e}", exc_info=True)
            return None

        # 2. 计算目标S3语音token（speech_token），用16kHz音频
        try:
            speech_tokens_batch, speech_token_lengths_batch = self.s3gen_model.tokenizer.forward(
                [wav_16k_tensor.numpy()], max_len=self.data_args.max_speech_token_len
            )
            if speech_tokens_batch is None or speech_token_lengths_batch is None: return None
            target_s3_tokens = speech_tokens_batch.squeeze(0) # (T_tokens_s3)
            speech_token_len = torch.tensor(target_s3_tokens.size(0), dtype=torch.long, device="cpu")
            # tokens与mel帧数量对齐（通常T_mel=2*T_tokens_s3）
            if target_mel.size(0) != 2 * target_s3_tokens.size(0) and target_mel.size(0) // 2 < target_s3_tokens.size(0):
                target_s3_tokens = target_s3_tokens[:target_mel.size(0)//2]
                speech_token_len = torch.tensor(target_s3_tokens.size(0), dtype=torch.long)
            elif target_mel.size(0) // 2 > target_s3_tokens.size(0) : # mel帧更长，token补零
                 pad_size = target_mel.size(0)//2 - target_s3_tokens.size(0)
                 target_s3_tokens = F.pad(target_s3_tokens, (0, pad_size), value=0)
        except Exception as e:
            logger.error(f"Item {idx}: Error tokenizing target speech: {e}", exc_info=True)
            return None

        # 3. 获取说话人嵌入（embedding），用16kHz音频
        model_device = next(self.s3gen_model.speaker_encoder.parameters()).device
        try:
            # 设置eval模式推理说话人向量
            original_training_state = self.s3gen_model.speaker_encoder.training
            self.s3gen_model.speaker_encoder.eval()

            speaker_embedding_batch = self.s3gen_model.speaker_encoder.inference(
                wav_16k_tensor.unsqueeze(0).to(model_device)
            )
            speaker_embedding = speaker_embedding_batch.squeeze(0)

        except Exception as e:
            logger.error(f"Item {idx}: Error getting speaker_embedding: {e}", exc_info=True)
            return None
        finally:
            if 'original_training_state' in locals():
                self.s3gen_model.speaker_encoder.train(original_training_state)

        # 4. 生成CausalMaskedDiffWithXvec调节所需的提示特征（prompt）
        prompt_wav_16k_segment = wav_16k_tensor[:self.prompt_audio_samples_16k]
        prompt_wav_24k_segment = wav_24k_tensor[:self.prompt_audio_samples_24k]

        if prompt_wav_16k_segment.numel() == 0 or prompt_wav_24k_segment.numel() == 0:
            # 音频过短时用全零填充
            max_flow_prompt_token_len = self.s3gen_model.flow.encoder.hp.get("prompt_token_max_len", 75)
            prompt_s3_tokens = torch.zeros(max_flow_prompt_token_len, dtype=torch.long)
            prompt_s3_token_len = torch.tensor(0, dtype=torch.long)
            prompt_mel = torch.zeros(max_flow_prompt_token_len * 2, target_mel.size(1), dtype=torch.float)
            prompt_mel_len = torch.tensor(0, dtype=torch.long)
        else:
            try:
                # 提示token
                max_flow_prompt_token_len = getattr(self.s3gen_model.flow.encoder, 'prompt_token_max_len', 75)
                prompt_s3_tokens_batch, prompt_s3_token_lengths_batch = self.s3gen_model.tokenizer.forward(
                    [prompt_wav_16k_segment.numpy()], max_len=max_flow_prompt_token_len
                )
                if prompt_s3_tokens_batch is None: return None
                prompt_s3_tokens = prompt_s3_tokens_batch.squeeze(0)
                prompt_s3_token_len = prompt_s3_token_lengths_batch.squeeze(0)

                # 提示mel谱
                prompt_mel = self.s3gen_model.mel_extractor(prompt_wav_24k_segment.unsqueeze(0)).squeeze(0).transpose(0,1)
                if prompt_mel.size(0) > prompt_s3_tokens.size(0) * 2:
                    prompt_mel = prompt_mel[:prompt_s3_tokens.size(0) * 2, :]
                prompt_mel_len = torch.tensor(prompt_mel.size(0), dtype=torch.long)
                if prompt_mel.size(0) // 2 < prompt_s3_tokens.size(0):
                    prompt_s3_tokens = prompt_s3_tokens[:prompt_mel.size(0)//2]
                    prompt_s3_token_len = torch.tensor(prompt_s3_tokens.size(0), dtype=torch.long)

            except Exception as e:
                logger.error(f"Item {idx}: Error processing prompt features: {e}", exc_info=True)
                return None

        return {
            "speech_token": target_s3_tokens.long(),        # 目标S3 token
            "speech_token_len": speech_token_len.long(),    # 目标S3 token长度
            "speech_feat": target_mel.float(),              # 目标梅尔谱
            "speech_feat_len": speech_feat_len.long(),      # 目标梅尔谱长度（帧数）
            "embedding": speaker_embedding.float(),         # 说话人嵌入
            "prompt_token_input": prompt_s3_tokens.long(),  # 提示S3 token
            "prompt_token_len_input": prompt_s3_token_len.long(),  # 提示S3 token长度
            "prompt_feat_input": prompt_mel.float(),        # 提示梅尔谱特征
        }


# --- S3Gen Data Collator ---
@dataclass
class S3GenDataCollator:
    """
    S3Gen批处理数据整理器，用于将单个样本pad并拼接为批量数据。
    主要用于PyTorch DataLoader的collate_fn，保证所有batch内的特征shape一致。
    """
    # pad用的数值。S3 token用0，梅尔谱可以设为0或-log(1e-5)等
    speech_token_pad_id: int = 0
    mel_pad_value: float = 0.0 # 根据flow模型需求，pad梅尔谱的值

    # 如果模型要求prompt长度固定，这里可配置最大prompt长度，否则默认按batch内最大长度pad
    # max_prompt_token_len_collator: Optional[int] = 75
    # max_prompt_mel_len_collator: Optional[int] = 150

    def __call__(self, features: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, Any]:
        """
        将多个样本pad并拼接成一个batch，供模型训练/推理用。
        features: 单样本组成的列表，每个元素是字典或None。
        返回：一个包含所有特征的字典，每个值都是(batch, ...)形状的Tensor。
        """
        valid_features = [f for f in features if f is not None]
        if not valid_features: return {}
        
        batch_size = len(valid_features)
        # 自动检测设备（通常是cpu或cuda）
        device = valid_features[0]["speech_token"].device if batch_size > 0 else "cpu"

        # 1. 对齐和pad speech_token (目标S3 token)
        speech_tokens = [f["speech_token"] for f in valid_features]
        max_len_st = max(s.size(0) for s in speech_tokens)
        padded_speech_tokens = torch.stack(
            [F.pad(s, (0, max_len_st - s.size(0)), value=self.speech_token_pad_id) for s in speech_tokens]
        )
        speech_token_lens = torch.stack([f["speech_token_len"] for f in valid_features])

        # 2. 对齐和pad speech_feat (目标梅尔谱，形状为(T_mel, F))
        speech_feats = [f["speech_feat"] for f in valid_features]
        max_len_sf = max(s.size(0) for s in speech_feats)
        mel_dim = speech_feats[0].size(1) # 梅尔谱特征维数
        padded_speech_feats = torch.stack(
            [F.pad(s, (0, 0, 0, max_len_sf - s.size(0)), value=self.mel_pad_value) for s in speech_feats]
        ) # 结果(B, T_mel_max, F)
        speech_feat_lens = torch.stack([f["speech_feat_len"] for f in valid_features])
        
        # 3. 说话人嵌入直接拼接(batch, D_spk)
        embeddings = torch.stack([f["embedding"] for f in valid_features])

        # 4. 对齐和pad prompt特征
        prompt_tokens = [f["prompt_token_input"] for f in valid_features]
        target_prompt_token_len = max(pt.size(0) for pt in prompt_tokens) if prompt_tokens else 0
        target_prompt_token_len = max(1, target_prompt_token_len) # 防止为0

        padded_prompt_tokens = torch.stack(
             [F.pad(pt, (0, target_prompt_token_len - pt.size(0)), value=self.speech_token_pad_id) for pt in prompt_tokens]
        )
        prompt_token_lens = torch.stack([f["prompt_token_len_input"] for f in valid_features])

        prompt_feats = [f["prompt_feat_input"] for f in valid_features]
        target_prompt_mel_len = max(pf.size(0) for pf in prompt_feats) if prompt_feats else 0
        target_prompt_mel_len = max(1, target_prompt_mel_len)
        
        padded_prompt_feats = torch.stack(
            [F.pad(pf, (0, 0, 0, target_prompt_mel_len - pf.size(0)), value=self.mel_pad_value) for pf in prompt_feats]
        )
        # prompt_feat_lens 可由prompt_feat的shape直接得到，通常不用显式传递

        # 构造返回batch字典，供flow模型训练forward使用
        return {
            'speech_token': padded_speech_tokens.to(device),        # (B, T_token)
            'speech_token_len': speech_token_lens.to(device),       # (B,)
            'speech_feat': padded_speech_feats.to(device),          # (B, T_mel, F)
            'speech_feat_len': speech_feat_lens.to(device),         # (B,)
            'embedding': embeddings.to(device),                     # (B, D_spk)
            "prompt_token_input": padded_prompt_tokens.to(device),  # (B, T_prompt_token)
            "prompt_token_len_input": prompt_token_lens.to(device), # (B,)
            "prompt_feat_input": padded_prompt_feats.to(device),    # (B, T_prompt_mel, F)
            # "prompt_feat_len_input": prompt_feat_lens.to(device) # 通常不显式返回
        }


# --- Model Wrapper for S3Gen's Flow component ---
class S3GenFlowForFineTuning(torch.nn.Module):
    """
    S3Gen Flow部分的微调封装模型。
    该类包装了S3Gen的flow模块，使其能与HuggingFace Trainer等生态无缝对接，支持finetune任务。
    """
    def __init__(self, s3gen_token_to_mel: S3Token2Mel): # Pass S3Token2Mel instance
        """
        初始化模型，提取flow模型和兼容HF的config信息。
        s3gen_token_to_mel: S3Token2Mel实例，里面包含flow等核心结构。
        """
        super().__init__()
        #self.s3_token_to_mel = s3gen_token_to_mel
        self.flow_model: CausalMaskedDiffWithXvec = s3gen_token_to_mel.flow # type: ignore
        
        # 创建一个伪HF config，保证Trainer等工具不会报错
        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_s3gen_flow_finetune"
            # 可根据需要补充S3Gen相关超参数
            def __init__(self, **kwargs): super().__init__(**kwargs)
        self.config = HFCompatibleConfig()
        # 如有需要，可以将s3_token_to_mel或flow_model里的参数同步到self.config

    def forward(self,
                speech_token: torch.Tensor,
                speech_token_len: torch.Tensor,
                speech_feat: torch.Tensor,          # Target mel (B, T_target_mel_collator, F)
                speech_feat_len: torch.Tensor,
                embedding: torch.Tensor,
                prompt_token_input: torch.Tensor,
                prompt_token_len_input: torch.Tensor,
                prompt_feat_input: torch.Tensor,    # Prompt mel (B, T_prompt_mel_collator, F)
                labels = None
                ):
        """
        前向传播，计算CFM loss。
        输入为DataCollator拼batch后的各种特征。
        返回：(loss, loss, 0.0)三元组。只用到第一个loss即可。
        """
        # 1. Project speaker embedding（投影说话人嵌入）
        projected_speaker_emb = self.flow_model.spk_embed_affine_layer(F.normalize(embedding, dim=1))

        # 2. 处理mu（语音内容编码，拼接prompt与target token，编码后取目标片段）
        full_input_tokens = torch.cat([prompt_token_input, speech_token], dim=1)
        full_input_token_lens = prompt_token_len_input + speech_token_len
        vocab_size = getattr(self.flow_model, 'vocab_size', 6561)
        input_token_embed = self.flow_model.input_embedding(
            torch.clamp(full_input_tokens, min=0, max=vocab_size - 1)
        )
        h_encoded_full, _ = self.flow_model.encoder(input_token_embed, full_input_token_lens)
        h_projected_full = self.flow_model.encoder_proj(h_encoded_full)
        mu_full_for_cfm = h_projected_full.transpose(1, 2).contiguous()

        # 3. 转置目标梅尔谱（CFM decoder用(B, F, T)格式）
        target_mels_for_cfm_x = speech_feat.transpose(1, 2).contiguous() # (B, F, T_target_mel_collator)
        target_mel_mask = (~self.make_pad_mask(speech_feat_len, max_len=target_mels_for_cfm_x.size(2))).unsqueeze(1)

        # 4. 准备prompt的梅尔谱，pad到和target一样长（后面补0表示“不调节”）
        original_prompt_mels = prompt_feat_input.transpose(1, 2).contiguous()
        
        T_target_mel_collator = target_mels_for_cfm_x.size(2)
        T_prompt_mel_collator = original_prompt_mels.size(2)

        if T_prompt_mel_collator > T_target_mel_collator:
            padded_prompt_mels_for_cond = original_prompt_mels[:, :, :T_target_mel_collator]
            logger.warning(f"Prompt mel length ({T_prompt_mel_collator}) was longer than target mel length ({T_target_mel_collator}). Prompt was truncated.")
        else:
            padding_size = T_target_mel_collator - T_prompt_mel_collator
            padded_prompt_mels_for_cond = F.pad(original_prompt_mels, (0, padding_size), mode='constant', value=0)
        # padded_prompt_mels_for_cond: (B, F, T_target_mel_collator)

        # 5. 从mu_full中切出与目标mel对齐的部分
        num_prompt_mel_frames_collator = T_prompt_mel_collator
        num_target_mel_frames_collator = T_target_mel_collator

        slice_start_idx = num_prompt_mel_frames_collator

        if slice_start_idx >= mu_full_for_cfm.size(2):
            logger.error(f"MU SLICING ERROR: mu_full_for_cfm (len {mu_full_for_cfm.size(2)}) too short for prompt part (len {slice_start_idx})")
            mu_conditioning_for_target_mels = torch.zeros_like(target_mels_for_cfm_x.float().expand(-1, mu_full_for_cfm.size(1), -1))
        else:
            mu_target_raw_slice = mu_full_for_cfm[:, :, slice_start_idx:]
            current_raw_target_slice_len = mu_target_raw_slice.size(2)
            if current_raw_target_slice_len < num_target_mel_frames_collator:
                padding_needed = num_target_mel_frames_collator - current_raw_target_slice_len
                mu_conditioning_for_target_mels = F.pad(mu_target_raw_slice, (0, padding_needed))
            elif current_raw_target_slice_len > num_target_mel_frames_collator:
                mu_conditioning_for_target_mels = mu_target_raw_slice[:, :, :num_target_mel_frames_collator]
            else:
                mu_conditioning_for_target_mels = mu_target_raw_slice

        # 6. 计算CFM（Conditional Flow Matching）损失
        cfm_loss_output, _ = self.flow_model.decoder.compute_loss(
            x1=target_mels_for_cfm_x,
            mask=target_mel_mask,
            mu=mu_conditioning_for_target_mels,
            spks=projected_speaker_emb,
            cond=padded_prompt_mels_for_cond
        )

        main_loss = cfm_loss_output
        return (main_loss, main_loss, torch.tensor(0.0, device=main_loss.device))

    def make_pad_mask(self, lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """
        生成batch内pad mask。返回形状为(batch, max_len)的bool张量。True表示pad，False表示有效帧。
        """
        # ... (same as before) ...
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.as_tensor(lengths, dtype=torch.long)
        if max_len is None:
            if lengths.numel() == 0: # Handle empty lengths tensor
                max_len = 0
            else:
                max_len = torch.max(lengths).item()

        bs = lengths.size(0)
        if bs == 0: # Handle batch size of 0
             return torch.empty(0, max_len, dtype=torch.bool, device=lengths.device)

        seq_range = torch.arange(0, max_len, device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand >= seq_length_expand

# 全局Trainer实例，用于控制训练
trainer_instance: Optional[Trainer] = None


def main():
    """
    S3Gen Flow部分微调的主入口。
    负责参数解析、模型加载、组件冻结、数据集加载、Trainer初始化、训练与评估等全流程。
    """
    global trainer_instance
    parser = HfArgumentParser((S3GenModelArguments, S3GenDataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    # --- 加载基础 ChatterboxTTS 模型以提取 S3Gen 部件 ---
    logger.info("Loading base ChatterboxTTS model to extract S3Gen components...")
    # 先加载完整模型，然后用于finetune（与T3微调流程类似）
    if model_args.local_model_dir:
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=model_args.local_model_dir, device="cpu")
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID # 可配置默认repo id
        download_dir = Path(training_args.output_dir) / "pretrained_chatterbox_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download
        files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
        for f in files_to_download:
            try: hf_hub_download(repo_id=repo_to_download, filename=f, local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
            except Exception as e: logger.warning(f"Could not download {f} from {repo_to_download}: {e}")
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=download_dir, device="cpu")

    s3gen_model_to_finetune: S3Gen = chatterbox_model.s3gen
    s3gen_token_to_mel_part: S3Token2Mel = s3gen_model_to_finetune # S3Gen继承关系保证了这一点

    # --- 冻结S3Gen的部分结构（只训练flow部分） ---
    # 冻结HiFiGAN部分（mel2wav）
    for param in s3gen_token_to_mel_part.mel2wav.parameters():
        param.requires_grad = False
    logger.info("S3Gen HiFiGAN (mel2wav) part frozen.")

    if model_args.freeze_speaker_encoder:
        for param in s3gen_token_to_mel_part.speaker_encoder.parameters():
            param.requires_grad = False
        logger.info("S3Gen Speaker Encoder (CAMPPlus) frozen.")
    
    if model_args.freeze_s3_tokenizer:
        # S3Tokenizer若有参数也冻结
        if hasattr(s3gen_token_to_mel_part.tokenizer, 'parameters'):
             for param in s3gen_token_to_mel_part.tokenizer.parameters(): # type: ignore
                param.requires_grad = False
        logger.info("S3Gen S3Tokenizer frozen (if it has parameters).")

    # 保证flow部分可训练
    for param in s3gen_token_to_mel_part.flow.parameters():
        param.requires_grad = True
    logger.info("S3Gen Flow Model (CausalMaskedDiffWithXvec) set to trainable.")

    # --- 加载和准备数据集 ---
    logger.info("Loading and processing dataset for S3Gen finetuning...")
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    if data_args.dataset_name:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name,
            cache_dir=model_args.cache_dir, verification_mode=verification_mode
        )
        train_hf_dataset = raw_datasets[data_args.train_split_name]
        eval_hf_dataset = raw_datasets.get(data_args.eval_split_name) if data_args.eval_split_name else None
        # 如果需要eval且没有eval分割，则自动按比例划分
        if training_args.do_eval and not eval_hf_dataset and data_args.eval_split_size > 0 and len(train_hf_dataset) > 1:
            split_dataset = train_hf_dataset.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train_hf_dataset, eval_hf_dataset = split_dataset["train"], split_dataset["test"]
    else:
        raise ValueError("S3Gen finetuning currently requires a Hugging Face dataset_name.")

    train_dataset = S3GenFineTuningDataset(data_args, s3gen_model_to_finetune, train_hf_dataset)
    eval_dataset = None
    if eval_hf_dataset and training_args.do_eval:
        eval_dataset = S3GenFineTuningDataset(data_args, s3gen_model_to_finetune, eval_hf_dataset)

    # --- Data Collator ---
    data_collator = S3GenDataCollator() # 默认填充设置

    # --- Trainer包装模型 ---
    s3gen_flow_trainable_model = S3GenFlowForFineTuning(s3gen_token_to_mel_part)

    # --- 指标计算函数 ---
    def compute_metrics_s3gen(eval_preds):
        """
        评估阶段回调，用于汇总loss到metrics。
        """
        metrics = {}
        if isinstance(eval_preds.predictions, tuple) and len(eval_preds.predictions) >= 1:
            if len(eval_preds.predictions) > 1 and eval_preds.predictions[1] is not None:
                metrics["eval_cfm_loss"] = float(np.mean(eval_preds.predictions[1]))
            if len(eval_preds.predictions) > 2 and eval_preds.predictions[2] is not None:
                metrics["eval_reg_loss"] = float(np.mean(eval_preds.predictions[2]))
        return metrics

    # --- 回调配置 ---
    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    # 如需音频生成callback，可扩展此处

    # --- Trainer实例化 ---
    logger.info(f"Using dataloader_pin_memory: {training_args.dataloader_pin_memory}")
    trainer_instance = Trainer(
        model=s3gen_flow_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_s3gen if training_args.do_eval and eval_dataset else None,
        callbacks=callbacks if callbacks else None
    )
    if training_args.label_names is None: trainer_instance.label_names = [] # 由模型自身处理target

    # --- 正式训练 ---
    if training_args.do_train:
        logger.info("*** Finetuning S3Gen Flow Model ***")
        train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer_instance.save_model() # 保存Trainer封装的模型
        
        logger.info("Saving finetuned S3Gen (flow part) model weights for ChatterboxTTS...")
        # 保存为Chatterbox格式的权重文件（.safetensors）
        finetuned_s3gen_state_dict = s3gen_model_to_finetune.state_dict()
        output_s3gen_safetensor_path = Path(training_args.output_dir) / "s3gen.safetensors"
        from safetensors.torch import save_file
        save_file(finetuned_s3gen_state_dict, output_s3gen_safetensor_path)
        logger.info(f"Finetuned S3Gen model weights saved to {output_s3gen_safetensor_path}")

        # 如需完整本地模型目录，可以复制其他必要文件
        # 见原英文注释
        
        metrics = train_result.metrics
        trainer_instance.log_metrics("train", metrics)
        trainer_instance.save_metrics("train", metrics)
        trainer_instance.save_state()

    # --- 评估 ---
    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating S3Gen Flow Model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)

    logger.info("S3Gen finetuning script finished.")

if __name__ == "__main__":
    main()
