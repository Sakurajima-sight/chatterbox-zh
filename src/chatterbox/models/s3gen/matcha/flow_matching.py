from abc import ABC

import torch
import torch.nn.functional as F

from .decoder import Decoder


class BASECFM(torch.nn.Module, ABC):
    """
    BASECFM 类用于实现基于条件流匹配（Conditional Flow Matching, CFM）的基础框架。
    该类包含前向传播、欧拉求解器和损失计算的实现，通常用于生成模型的训练和推理阶段。
    """

    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        """
        初始化BASECFM类。

        参数:
        - n_feats: 特征的数量（如mel频谱图的频率数）
        - cfm_params: CFM的相关参数配置
        - n_spks: 说话人的数量（默认为1）
        - spk_emb_dim: 说话人嵌入的维度（默认为128）
        """
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver

        # 如果参数中没有sigma_min，使用默认值1e-4
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """
        前向扩散过程。

        参数:
        - mu (torch.Tensor): 编码器输出，形状为 (batch_size, n_feats, mel_timesteps)
        - mask (torch.Tensor): 输出的mask，形状为 (batch_size, 1, mel_timesteps)
        - n_timesteps (int): 扩散步骤数
        - temperature (float, optional): 用于缩放噪声的温度，默认为1.0
        - spks (torch.Tensor, optional): 说话人ID，形状为 (batch_size, spk_emb_dim)，默认为None
        - cond (optional): 目前未使用，但为将来用途保留

        返回:
        - sample (torch.Tensor): 生成的mel频谱图，形状为 (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature  # 生成随机噪声
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)  # 时间跨度
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        欧拉求解器用于解ODE（常微分方程）。
        
        参数:
        - x (torch.Tensor): 随机噪声
        - t_span (torch.Tensor): n_timesteps插值的时间跨度，形状为 (n_timesteps + 1,)
        - mu (torch.Tensor): 编码器输出，形状为 (batch_size, n_feats, mel_timesteps)
        - mask (torch.Tensor): 输出的mask，形状为 (batch_size, 1, mel_timesteps)
        - spks (torch.Tensor, optional): 说话人嵌入，形状为 (batch_size, spk_emb_dim)
        - cond (optional): 目前未使用，但为将来用途保留
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            x = x + dt * dphi_dt  # 使用欧拉方法更新x
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """
        计算扩散过程中的损失。

        参数:
        - x1 (torch.Tensor): 目标，形状为 (batch_size, n_feats, mel_timesteps)
        - mask (torch.Tensor): 目标mask，形状为 (batch_size, 1, mel_timesteps)
        - mu (torch.Tensor): 编码器输出，形状为 (batch_size, n_feats, mel_timesteps)
        - spks (torch.Tensor, optional): 说话人嵌入，形状为 (batch_size, spk_emb_dim)

        返回:
        - loss: 条件流匹配损失
        - y: 条件流，形状为 (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)  # 随机时间步
        z = torch.randn_like(x1)  # 生成噪声

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1  # 条件流匹配的y
        u = x1 - (1 - self.sigma_min) * z  # 计算u

        # 使用MSE损失计算条件流匹配损失
        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


class CFM(BASECFM):
    """
    CFM 类继承自BASECFM，实现了基于条件流匹配（CFM）的特定框架。
    该类的估计器部分采用了解码器（Decoder）结构，用于生成和恢复mel频谱图。
    """

    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        """
        初始化CFM类，配置解码器和其他必要的参数。

        参数:
        - in_channels: 输入通道数
        - out_channel: 输出通道数
        - cfm_params: CFM的相关参数
        - decoder_params: 解码器的相关参数
        - n_spks: 说话人数量（默认为1）
        - spk_emb_dim: 说话人嵌入维度（默认为64）
        """
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)  # 根据说话人嵌入调整输入通道数
        # 使用解码器作为估计器进行条件流匹配
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
