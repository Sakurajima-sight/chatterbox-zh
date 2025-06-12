import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from .utils.mask import make_pad_mask


class MaskedDiffWithXvec(torch.nn.Module):
    """
    该类实现了基于X-Vector的有遮罩扩散模型。它结合了文本信息、说话人嵌入（X-Vec）和mel频谱图，用于生成语音特征。
    该模型通过融合文本特征和语音特征进行训练和推理，支持条件生成任务。
    """

    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        """
        初始化MaskedDiffWithXvec模型。

        参数：
            input_size: 输入特征的大小（默认为512）
            output_size: 输出特征的大小（默认为80）
            spk_embed_dim: 说话人嵌入的维度（默认为192）
            output_type: 输出类型（默认为"mel"）
            vocab_size: 词汇表大小（默认为4096）
            input_frame_rate: 输入帧率（默认为50）
            only_mask_loss: 是否只计算遮罩损失（默认为True）
            encoder: 编码器模块，用于处理输入的文本
            length_regulator: 长度调节器模块，用于调节音频长度
            decoder: 解码器模块，用于生成语音特征
            decoder_conf: 解码器配置字典，包含多个参数
            mel_feat_conf: Mel频谱图配置字典
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)  # 词嵌入层
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)  # 说话人嵌入映射
        self.encoder = encoder  # 文本编码器
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)  # 编码器输出映射
        self.decoder = decoder  # 解码器
        self.length_regulator = length_regulator  # 长度调节器
        self.only_mask_loss = only_mask_loss  # 是否只计算遮罩损失

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        前向传播函数，处理输入的batch数据并返回损失。

        参数：
            batch: 输入的批次数据，包含文本token、文本长度、特征、特征长度和说话人嵌入
            device: 当前计算设备（CPU或GPU）

        返回：
            返回损失字典，包含损失值
        """
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # xvec投影
        embedding = F.normalize(embedding, dim=1)  # 对说话人嵌入进行归一化
        embedding = self.spk_embed_affine_layer(embedding)  # 线性变换为输出特征

        # 拼接文本和提示文本
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  # 获取文本嵌入并应用mask

        # 文本编码
        h, h_lengths = self.encoder(token, token_len)  # 通过编码器处理文本
        h = self.encoder_proj(h)  # 投影为输出维度
        h, h_lengths = self.length_regulator(h, feat_len)  # 使用长度调节器调整特征长度

        # 获取条件
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))  # 随机生成部分条件
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)  # 转置为适合的形状

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)  # 上采样特征
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        """
        推理函数，生成mel频谱图。

        参数：
            token: 输入的文本token
            token_len: 输入文本的长度
            prompt_token: 提示文本token
            prompt_token_len: 提示文本长度
            prompt_feat: 提示特征
            prompt_feat_len: 提示特征长度
            embedding: 说话人嵌入
            flow_cache: 流缓存，用于加速推理

        返回：
            feat: 生成的mel频谱图
            flow_cache: 更新后的流缓存
        """
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()  # 将特征转换为半精度
            embedding = embedding.half()  # 将嵌入转换为半精度

        assert token.shape[0] == 1  # 确保输入批次大小为1
        # xvec投影
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # 拼接文本和提示文本
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # 文本编码
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # 获取条件
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]  # 去掉提示部分
        assert feat.shape[2] == mel_len2  # 确保输出形状正确
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    """
    该类实现了基于X-Vector的有遮罩因果扩散模型（Causal Masked Diffusion with X-Vec）。
    该模型将文本和说话人嵌入（X-Vec）结合起来，用于生成mel频谱图，支持条件生成任务。
    通过因果扩散过程，模型在给定输入的情况下生成输出特征。
    """

    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 6561,
                 input_frame_rate: int = 25,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        """
        初始化CausalMaskedDiffWithXvec模型。

        参数：
            input_size: 输入特征的大小（默认为512）
            output_size: 输出特征的大小（默认为80）
            spk_embed_dim: 说话人嵌入的维度（默认为192）
            output_type: 输出类型（默认为"mel"）
            vocab_size: 词汇表大小（默认为6561）
            input_frame_rate: 输入帧率（默认为25）
            only_mask_loss: 是否只计算遮罩损失（默认为True）
            token_mel_ratio: 文本与mel频谱的比率（默认为2）
            pre_lookahead_len: 提前预处理的长度（默认为3）
            encoder: 编码器模块，用于处理输入的文本
            decoder: 解码器模块，用于生成语音特征
            decoder_conf: 解码器配置字典，包含多个参数
            mel_feat_conf: Mel频谱图配置字典
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)  # 词嵌入层
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)  # 说话人嵌入映射
        self.encoder = encoder  # 文本编码器
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)  # 编码器输出映射
        self.decoder = decoder  # 解码器
        self.only_mask_loss = only_mask_loss  # 是否只计算遮罩损失
        self.token_mel_ratio = token_mel_ratio  # 文本与mel频谱的比率
        self.pre_lookahead_len = pre_lookahead_len  # 提前处理的长度

        # FIXME: 该值目前为False，后续可以修改
        self.fp16 = False  # 是否使用16位浮点数精度

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        """
        推理函数，生成mel频谱图。

        参数：
            token: 输入的文本token
            token_len: 输入文本的长度
            prompt_token: 提示文本token
            prompt_token_len: 提示文本长度
            prompt_feat: 提示特征
            prompt_feat_len: 提示特征长度
            embedding: 说话人嵌入
            finalize: 是否为最终步骤的标志

        返回：
            feat: 生成的mel频谱图
            None: 返回None作为占位符
        """
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()  # 将特征转换为半精度
            embedding = embedding.half()  # 将嵌入转换为半精度

        assert token.shape[0] == 1  # 确保输入批次大小为1
        # xvec投影
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # 拼接文本和提示文本
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # 文本编码
        h, h_lengths = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]  # 如果不是最终步骤，截断部分h
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)  # 投影为输出维度

        # 获取条件
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat  # 提示特征
        conds = conds.transpose(1, 2)  # 转置为适合的形状

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10  # 设定扩散的步数
        )
        feat = feat[:, :, mel_len1:]  # 去掉提示部分
        assert feat.shape[2] == mel_len2  # 确保输出形状正确
        return feat.float(), None  # 返回生成的特征和None（占位符）
