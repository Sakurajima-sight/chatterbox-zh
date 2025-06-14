# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import torch
import torch.nn.functional as F
from .matcha.flow_matching import BASECFM
from omegaconf import OmegaConf


# 配置参数，包括噪声的最小标准差、时间调度器、训练和推理的配置率等
CFM_PARAMS = OmegaConf.create({
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
})

class ConditionalCFM(BASECFM):
    """
    条件流匹配模型，继承自BASECFM，主要用于条件生成任务，包含扩散模型的前向传播、解算器和损失计算。
    """

    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        """
        初始化条件流匹配模型的各个组件。

        参数：
            in_channels: 输入特征通道数
            cfm_params: 配置参数
            n_spks: 说话人数量，默认为1
            spk_emb_dim: 说话人嵌入的维度，默认为64
            estimator: 用于估计的模块（默认为None）
        """
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        self.estimator = estimator  # 用于估计的模块
        self.lock = threading.Lock()  # 用于多线程保护

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, flow_cache=torch.zeros(1, 80, 0, 2)):
        """
        执行扩散过程的前向传播。

        参数：
            mu (torch.Tensor): 编码器输出，形状为(batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): 输出mask，形状为(batch_size, 1, mel_timesteps)
            n_timesteps (int): 扩散步骤数
            temperature (float, optional): 用于调节噪声的温度，默认为1.0
            spks (torch.Tensor, optional): 说话人嵌入，默认为None
            cond: 未使用，但保留以便将来扩展

        返回：
            sample: 生成的mel频谱图，形状为(batch_size, n_feats, mel_timesteps)
            flow_cache: 存储的流缓存，用于后续的步骤
        """
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = flow_cache.shape[2]
        
        # 修复提示和重叠部分
        if cache_size != 0:
            z[:, :, :cache_size] = flow_cache[:, :, :, 0]
            mu[:, :, :cache_size] = flow_cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        flow_cache = torch.stack([z_cache, mu_cache], dim=-1)

        # 时间跨度
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), flow_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        用于解算扩散方程的欧拉方法。

        参数：
            x (torch.Tensor): 随机噪声
            t_span (torch.Tensor): 插值后的时间跨度，形状为(n_timesteps + 1,)
            mu (torch.Tensor): 编码器输出，形状为(batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): 输出mask，形状为(batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): 说话人嵌入，默认为None
            cond: 未使用，但保留以便将来扩展

        返回：
            sol: 扩散方程的最终解
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        
        for step in range(1, len(t_span)):
            # 计算Classifier-Free Guidance（CFG）
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        """
        通过估计器进行前向传播计算。

        参数：
            x, mask, mu, t, spks, cond: 输入数据
        
        返回：
            估计器的输出
        """
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator.forward(x, mask, mu, t, spks, cond)
        else:
            with self.lock:
                # 设置输入形状
                self.estimator.set_input_shape('x', (2, 80, x.size(2)))
                self.estimator.set_input_shape('mask', (2, 1, x.size(2)))
                self.estimator.set_input_shape('mu', (2, 80, x.size(2)))
                self.estimator.set_input_shape('t', (2,))
                self.estimator.set_input_shape('spks', (2, 80))
                self.estimator.set_input_shape('cond', (2, 80, x.size(2)))
                # 运行 TRT 引擎
                self.estimator.execute_v2([x.contiguous().data_ptr(),
                                           mask.contiguous().data_ptr(),
                                           mu.contiguous().data_ptr(),
                                           t.contiguous().data_ptr(),
                                           spks.contiguous().data_ptr(),
                                           cond.contiguous().data_ptr(),
                                           x.data_ptr()])
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """
        计算扩散损失

        参数：
            x1 (torch.Tensor): 目标张量，形状为(batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): 目标mask，形状为(batch_size, 1, mel_timesteps)
            mu (torch.Tensor): 编码器输出，形状为(batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): 说话人嵌入，默认为None
            cond (torch.Tensor, optional): 条件张量，默认为None

        返回：
            loss: 条件流匹配损失
            y: 条件流，形状为(batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # 随机时间步
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # 采样噪声
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # 在训练过程中，我们随机丢弃条件，以平衡模式覆盖和样本逼真度
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    """
    该类实现了一个基于条件流匹配的因果扩散模型，继承自ConditionalCFM类。
    它通过添加随机噪声和调整扩散过程，生成mel频谱图，特别适用于时序数据的生成任务。
    """

    def __init__(self, in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=None):
        """
        初始化CausalConditionalCFM模型。

        参数：
            in_channels: 输入特征的通道数（默认为240）
            cfm_params: 配置参数，控制扩散过程等（默认为CFM_PARAMS）
            n_spks: 说话人数量（默认为1）
            spk_emb_dim: 说话人嵌入的维度（默认为80）
            estimator: 估计器，默认为None
        """
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        # 随机噪声，用于扩散过程的初始化
        self.rand_noise = torch.randn([1, 80, 50 * 300])  # 生成随机噪声，形状为(1, 80, 15000)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """
        执行扩散过程的前向传播。

        参数：
            mu (torch.Tensor): 编码器输出，形状为(batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): 输出mask，形状为(batch_size, 1, mel_timesteps)
            n_timesteps (int): 扩散步骤数
            temperature (float, optional): 用于调节噪声的温度，默认为1.0
            spks (torch.Tensor, optional): 说话人嵌入，默认为None，形状为(batch_size, spk_emb_dim)
            cond: 未使用，但保留以便将来扩展

        返回：
            sample: 生成的mel频谱图，形状为(batch_size, n_feats, mel_timesteps)
            None: 目前未使用，保留以便将来扩展
        """
        # 根据温度调整噪声，并确保噪声的尺寸与mu匹配
        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        
        # 时间跨度
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        
        # 使用欧拉方法求解扩散方程
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None
