# jrm: adapted from CosyVoice/cosyvoice/hifigan/generator.py
#      most modules should be reusable, but I found their SineGen changed a git.

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
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

"""HIFI-GAN"""

from typing import Dict, Optional, List
import numpy as np
from scipy.signal import get_window
import torch
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.distributions.uniform import Uniform
from torch import nn, sin, pow
from torch.nn import Parameter


class Snake(nn.Module):
    """
    实现了一种基于正弦的周期性激活函数（Snake）。
    该激活函数的输出与输入具有相同的形状(B, C, T)。
    
    参数：
        - alpha: 可训练的参数，决定了正弦函数的频率。默认值为1.0。
        - alpha_trainable: 是否允许alpha参数训练，默认为True。
        - alpha_logscale: 如果为True，则alpha将在对数尺度上训练，默认为False。

    参考文献：
        该激活函数来自Liu Ziyin, Tilman Hartwig, Masahito Ueda的论文：
        https://arxiv.org/abs/2006.08195

    示例：
        >>> a1 = Snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        初始化Snake激活函数的参数。

        参数：
            - in_features: 输入特征的形状
            - alpha: 可训练参数，控制正弦频率，默认值为1.0
            - alpha_trainable: 是否训练alpha，默认为True
            - alpha_logscale: 是否使用对数尺度训练alpha，默认为False
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # 初始化alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # 如果alpha是对数尺度初始化，则初始化为0
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # 如果是线性尺度初始化，则初始化为1
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001  # 防止除零错误

    def forward(self, x):
        """
        前向传播：应用正弦激活函数。

        参数：
            - x: 输入张量，形状为(B, C, T)

        返回：
            - x: 激活后的输出张量
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # 调整alpha的形状，使其与x对齐
        if self.alpha_logscale:
            alpha = torch.exp(alpha)  # 如果alpha是对数尺度，计算其指数
        # 计算Snake激活函数： x + 1/alpha * sin^2(x * alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


def get_padding(kernel_size, dilation=1):
    """
    获取卷积操作的padding大小。

    参数：
        - kernel_size: 卷积核的大小
        - dilation: 膨胀因子，默认为1

    返回：
        - padding: 计算得到的padding大小
    """
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    """
    初始化权重函数。用于卷积层和其他需要初始化的层。

    参数：
        - m: 模型中的模块
        - mean: 权重初始化的均值，默认为0.0
        - std: 权重初始化的标准差，默认为0.01
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:  # 如果是卷积层
        m.weight.data.normal_(mean, std)  # 使用正态分布初始化卷积层的权重


"""hifigan based generator implementation.

This code is modified from https://github.com/jik876/hifi-gan
 ,https://github.com/kan-bayashi/ParallelWaveGAN and
 https://github.com/NVIDIA/BigVGAN

"""


class ResBlock(torch.nn.Module):
    """
    HiFiGAN/BigVGAN中的残差模块（Residual Block）。
    该模块用于对输入特征进行处理，并通过残差连接保留特征信息。
    """

    def __init__(self, channels: int = 512, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        """
        初始化ResBlock模块。

        参数：
            channels: 输入和输出的通道数（默认为512）
            kernel_size: 卷积核的大小（默认为3）
            dilations: 膨胀率（默认为[1, 3, 5]），用于改变卷积感受野
        """
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList()  # 第一个卷积层列表
        self.convs2 = nn.ModuleList()  # 第二个卷积层列表

        for dilation in dilations:
            # 对每个膨胀率进行卷积操作
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation)
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1)
                    )
                )
            )
        # 对卷积层进行初始化
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        
        # 为每个卷积层添加激活函数
        self.activations1 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入进行处理，通过每一层的卷积和激活函数进行操作。

        参数：
            x: 输入张量，形状为(B, C, T)

        返回：
            x: 经过处理的输出张量，形状为(B, C, T)
        """
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)  # 应用激活函数
            xt = self.convs1[idx](xt)  # 应用第一个卷积
            xt = self.activations2[idx](xt)  # 应用第二个激活函数
            xt = self.convs2[idx](xt)  # 应用第二个卷积
            x = xt + x  # 残差连接
        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])


class SineGen(torch.nn.Module):
    """
    正弦波生成器（Sine Generator）定义。

    参数：
        samp_rate: 采样率，单位Hz
        harmonic_num: 谐波数，默认为0
        sine_amp: 正弦波的振幅，默认为0.1
        noise_std: 高斯噪声的标准差，默认为0.003
        voiced_threshold: 用于U/V分类的F0阈值，默认为0
        flag_for_pulse: 指示是否用于脉冲生成器（默认为False）
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        """
        初始化正弦波生成器的参数。

        参数：
            samp_rate: 采样率，单位Hz
            harmonic_num: 谐波数，默认为0
            sine_amp: 正弦波的振幅，默认为0.1
            noise_std: 高斯噪声的标准差，默认为0.003
            voiced_threshold: 用于U/V分类的F0阈值，默认为0
        """
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        """
        生成U/V信号（语音或非语音）。
        根据F0值生成一个二值化的语音状态标记。

        参数：
            f0: 基频（F0）值

        返回：
            uv: 语音状态标记，1表示语音，0表示非语音
        """
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    @torch.no_grad()
    def forward(self, f0):
        """
        正弦波生成的前向传播。

        参数：
            f0: 基频（F0），形状为[B, 1, sample_len]，单位Hz

        返回：
            sine_waves: 生成的正弦波，形状为[B, 1, sample_len]
            uv: 语音状态标记，形状为[B, 1, sample_len]
            noise: 高斯噪声，形状为[B, 1, sample_len]
        """
        # 初始化频率矩阵F_mat，形状为[B, harmonic_num+1, sample_len]
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i: i + 1, :] = f0 * (i + 1) / self.sampling_rate  # 计算每个谐波的频率

        # 计算相位矩阵
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0  # 设置第一个谐波的相位为0

        # 生成正弦波
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)

        # 生成U/V信号
        uv = self._f02uv(f0)

        # 为非语音区域生成噪声（用于无声部分的噪声）
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # 对于无声部分，使用0；对于有声部分，加上噪声
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """
    hn-nsf模型中的SourceModule，负责生成基于F0的正弦波和噪声源信号。
    
    参数：
        sampling_rate: 采样率，单位为Hz
        harmonic_num: 基频F0之上的谐波数，默认为0
        sine_amp: 正弦波的振幅，默认为0.1
        add_noise_std: 高斯噪声的标准差，默认为0.003
            注意：无声区域的噪声幅度由sine_amp决定
        voiced_threshold: 用于根据F0设置U/V分类的阈值，默认为0
        
    返回：
        Sine_source, noise_source: 返回生成的正弦源和噪声源
        uv: 语音状态标记（语音/非语音），形状为(batchsize, length, 1)
    """
    
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        """
        初始化SourceModuleHnNSF模块。

        参数：
            sampling_rate: 采样率，单位为Hz
            upsample_scale: 上采样比例
            harmonic_num: 谐波数，默认为0
            sine_amp: 正弦波振幅，默认为0.1
            add_noise_std: 高斯噪声的标准差，默认为0.003
            voiced_threshod: 用于设置U/V的F0阈值，默认为0
        """
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # 创建用于生成正弦波形的SineGen模块
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # 用于将多个谐波源合并为一个单一的激励信号
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        前向传播，生成正弦波源、噪声源以及U/V分类信号。

        参数：
            x: 输入F0值，形状为(batchsize, length, 1)
        
        返回：
            sine_merge: 合并后的正弦波信号
            noise: 生成的噪声信号
            uv: 语音状态标记（语音/非语音），形状为(batchsize, length, 1)
        """
        # 生成谐波分支的源信号
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1, 2))  # 生成正弦波信号
            sine_wavs = sine_wavs.transpose(1, 2)
            uv = uv.transpose(1, 2)
        
        # 合并生成的正弦波信号
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))  # 合并后的正弦波信号

        # 生成噪声信号，噪声的形状与uv相同
        noise = torch.randn_like(uv) * self.sine_amp / 3  # 为非语音区域生成噪声
        
        return sine_merge, noise, uv


class HiFTGenerator(nn.Module):
    """
    HiFTNet生成器：神经源滤波器 + ISTFTNet
    参考文献: https://arxiv.org/abs/2309.09493

    该类实现了HiFT（Neural Source Filter）网络的生成器部分，结合了神经源滤波器（NSF）和逆短时傅里叶变换网络（ISTFTNet）进行语音生成。
    """

    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            nb_harmonics: int = 8,
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: List[int] = [8, 8],
            upsample_kernel_sizes: List[int] = [16, 16],
            istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: List[int] = [3, 7, 11],
            resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: List[int] = [7, 11],
            source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            f0_predictor: torch.nn.Module = None,
    ):
        """
        初始化HiFTGenerator生成器的各个模块和超参数。

        参数：
            in_channels: 输入通道数，默认为80
            base_channels: 基础通道数，默认为512
            nb_harmonics: 谐波数，默认为8
            sampling_rate: 采样率，默认为22050
            nsf_alpha: NSF中的alpha值，控制正弦源的幅度，默认为0.1
            nsf_sigma: NSF中的噪声标准差，默认为0.003
            nsf_voiced_threshold: 用于语音/非语音分类的F0阈值，默认为10
            upsample_rates: 上采样率，默认为[8, 8]
            upsample_kernel_sizes: 上采样卷积核大小，默认为[16, 16]
            istft_params: ISTFT参数，包含n_fft和hop_len等，默认为{"n_fft": 16, "hop_len": 4}
            resblock_kernel_sizes: 残差块的卷积核大小，默认为[3, 7, 11]
            resblock_dilation_sizes: 残差块的膨胀大小，默认为[[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            source_resblock_kernel_sizes: 来源的残差块卷积核大小，默认为[7, 11]
            source_resblock_dilation_sizes: 来源的残差块膨胀大小，默认为[[1, 3, 5], [1, 3, 5]]
            lrelu_slope: LeakyReLU的斜率，默认为0.1
            audio_limit: 输出音频的幅度限制，默认为0.99
            f0_predictor: 基频预测模块，默认为None
        """
        super(HiFTGenerator, self).__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # 初始化SourceModuleHnNSF模块（用于生成源信号）
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold)
        
        # 基频的上采样模块
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        # 预处理卷积层
        self.conv_pre = weight_norm(
            Conv1d(in_channels, base_channels, 7, 1, padding=3)
        )

        # 上采样部分
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # 源信号的下采样部分
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1)
                )
            else:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u, padding=(u // 2))
                )

            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d)
            )

        # 残差块部分
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        # 后处理卷积层
        self.conv_post = weight_norm(Conv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
        # 边界反射填充层
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        
        # 用于STFT的窗函数
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))

        # 基频预测器
        self.f0_predictor = f0_predictor

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        self.m_source.remove_weight_norm()
        for l in self.source_downs:
            remove_weight_norm(l)
        for l in self.source_resblocks:
            l.remove_weight_norm()

    def _stft(self, x):
        """
        执行短时傅里叶变换（STFT）。
        
        参数：
            x: 输入信号
        
        返回：
            spec: 频谱的实部和虚部
        """
        spec = torch.stft(
            x,
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to(x.device),
            return_complex=True)
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        """
        执行逆短时傅里叶变换（ISTFT）。

        参数：
            magnitude: 频谱幅度
            phase: 频谱相位
        
        返回：
            inverse_transform: 重建的信号
        """
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"],
                                        self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        """
        解码输入的特征，并生成语音信号。

        参数：
            x: 输入特征
            s: 输入的源信号

        返回：
            x: 生成的语音信号
        """
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # 融合
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # 实际上，sin是冗余的

        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        前向传播函数，接收输入批次数据，生成语音信号。
        
        参数：
            batch: 输入的批次数据，包含语音特征
            device: 当前设备（CPU/GPU）

        返回：
            generated_speech: 生成的语音信号
            f0: 预测的基频
        """
        speech_feat = batch['speech_feat'].transpose(1, 2).to(device)
        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        # mel+source->speech
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, f0

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        """
        推理函数，生成语音信号。

        参数：
            speech_feat: 输入的语音特征
            cache_source: 缓存的源信号，用于避免音频生成中的卡顿

        返回：
            generated_speech: 生成的语音信号
            s: 源信号
        """
        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        # 使用缓存源信号避免卡顿
        if cache_source.shape[2] != 0:
            s[:, :, :cache_source.shape[2]] = cache_source
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s
