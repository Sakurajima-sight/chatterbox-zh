#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)

from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio.compliance.kaldi as Kaldi


def pad_list(xs, pad_value):
    """对张量列表进行填充

    参数：
        xs (List): 张量列表 [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)]。
        pad_value (float): 填充值。

    返回：
        Tensor: 填充后的张量 (B, Tmax, `*)。

    示例：
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def extract_feature(audio):
    """提取音频特征

    参数：
        audio (List): 输入音频列表。

    返回：
        Tuple: 填充后的特征张量，特征的长度，以及每个音频的时间信息。
    """
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)  # 对特征进行均值归一化
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    # 对批量推理进行填充
    features_padded = pad_list(features, pad_value=0)
    return features_padded, feature_lengths, feature_times


class BasicResBlock(torch.nn.Module):
    """一个基本的残差块，用于构建神经网络的深层模块

    参数：
        in_planes (int): 输入通道数
        planes (int): 输出通道数
        stride (int): 步幅大小，默认为1
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                torch.nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积 + 激活函数
        out = self.bn2(self.conv2(out))  # 第二层卷积
        out += self.shortcut(x)  # 加上shortcut残差连接
        out = F.relu(out)  # 激活函数
        return out


class FCM(torch.nn.Module):
    """用于处理音频特征的卷积神经网络模型

    参数：
        block: 残差块类型，默认为 BasicResBlock
        num_blocks: 每一层的残差块数目，默认为[2, 2]
        m_channels: 中间卷积通道数，默认为32
        feat_dim: 输入特征维度，默认为80
    """
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = torch.nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = torch.nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        """构建网络层，堆叠多个残差块"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """前向传播函数

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 输出特征张量
        """
        x = x.unsqueeze(1)  # 增加一个维度来适应卷积层的输入
        out = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积 + 激活函数
        out = self.layer1(out)  # 第一层残差块
        out = self.layer2(out)  # 第二层残差块
        out = F.relu(self.bn2(self.conv2(out)))  # 第二层卷积 + 激活函数

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])  # 调整输出形状
        return out


def get_nonlinear(config_str, channels):
    """根据配置字符串构建非线性激活层

    参数：
        config_str (str): 配置字符串，表示非线性层的顺序
        channels (int): 激活函数的通道数

    返回：
        Sequential: 包含多个非线性操作的顺序层
    """
    nonlinear = torch.nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", torch.nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    """统计池化函数，计算输入张量的均值和标准差

    参数：
        x (Tensor): 输入张量
        dim (int): 按哪个维度进行池化，默认为最后一个维度
        keepdim (bool): 是否保留维度，默认为False
        unbiased (bool): 是否使用无偏估计，默认为True
        eps (float): 防止除零错误的极小值，默认为1e-2

    返回：
        Tensor: 拼接后的均值和标准差张量
    """
    mean = x.mean(dim=dim)  # 计算均值
    std = x.std(dim=dim, unbiased=unbiased)  # 计算标准差
    stats = torch.cat([mean, std], dim=-1)  # 拼接均值和标准差
    if keepdim:
        stats = stats.unsqueeze(dim=dim)  # 保持维度不变
    return stats


class StatsPool(torch.nn.Module):
    """统计池化层，将输入进行均值和标准差池化

    该类使用了statistics_pooling函数来实现对输入特征的池化操作。
    """

    def forward(self, x):
        """前向传播函数

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 统计池化后的特征张量
        """
        return statistics_pooling(x)


class TDNNLayer(torch.nn.Module):
    """时间卷积神经网络（TDNN）层，用于处理时序数据

    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核的大小
        stride (int): 卷积步幅，默认为1
        padding (int): 填充大小，默认为0
        dilation (int): 膨胀大小，默认为1
        bias (bool): 是否使用偏置，默认为False
        config_str (str): 配置字符串，指定非线性操作（例如 "batchnorm-relu"）
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super(TDNNLayer, self).__init__()
        # 确保填充大小正确
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        
        # 1D卷积层
        self.linear = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        
        # 非线性操作
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        """前向传播函数

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 输出特征张量
        """
        x = self.linear(x)  # 卷积操作
        x = self.nonlinear(x)  # 非线性激活
        return x


class CAMLayer(torch.nn.Module):
    """通道注意力模块（CAM）层，结合卷积操作和通道注意力机制

    参数：
        bn_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核的大小
        stride (int): 卷积步幅
        padding (int): 填充大小
        dilation (int): 膨胀大小
        bias (bool): 是否使用偏置
        reduction (int): 通道注意力的压缩比例，默认为2
    """
    def __init__(
        self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2
    ):
        super(CAMLayer, self).__init__()
        self.linear_local = torch.nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        # 通道注意力模块
        self.linear1 = torch.nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """前向传播函数

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 经过通道注意力机制后的输出
        """
        y = self.linear_local(x)  # 卷积操作
        # 计算上下文信息
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))  # 通道注意力权重
        return y * m  # 加权输出

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        """分段池化函数，可以选择平均池化或最大池化

        参数：
            x (Tensor): 输入特征张量
            seg_len (int): 每个分段的长度，默认为100
            stype (str): 池化类型，可以是"avg"或"max"

        返回：
            Tensor: 池化后的特征张量
        """
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(torch.nn.Module):
    """结合通道注意力机制和密集时间卷积层（CAMDenseTDNNLayer）

    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        bn_channels (int): 中间层的通道数
        kernel_size (int): 卷积核的大小
        stride (int): 步幅大小
        dilation (int): 膨胀大小
        bias (bool): 是否使用偏置
        config_str (str): 非线性激活函数配置字符串
        memory_efficient (bool): 是否使用内存高效的操作
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, "Expect equal paddings, but got even kernel size ({})".format(
            kernel_size
        )
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = torch.nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        """批归一化函数，用于输入特征

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 执行批归一化后的特征
        """
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        """前向传播函数

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 经过处理后的特征
        """
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)  # 使用检查点来节省内存
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))  # 经过CAM层处理
        return x


class CAMDenseTDNNBlock(torch.nn.ModuleList):
    """CAMDenseTDNNBlock类：由多个CAMDenseTDNNLayer层构成的模块，通过将每一层的输出与输入拼接在一起，形成一个更深的特征表示。

    参数：
        num_layers (int): 堆叠的CAMDenseTDNNLayer层的数量
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        bn_channels (int): 中间层的通道数
        kernel_size (int): 卷积核的大小
        stride (int): 步幅大小，默认为1
        dilation (int): 膨胀大小，默认为1
        bias (bool): 是否使用偏置，默认为False
        config_str (str): 非线性激活函数配置字符串，默认为"batchnorm-relu"
        memory_efficient (bool): 是否使用内存高效的操作，默认为False
    """
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            # 创建每一层的CAMDenseTDNNLayer，并将其加入模块
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        """前向传播函数，按顺序执行所有层并将它们的输出拼接在一起

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 拼接后的输出特征张量
        """
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)  # 拼接每一层的输出
        return x


class TransitLayer(torch.nn.Module):
    """TransitLayer类：用于在网络中改变通道数的卷积层，同时应用非线性激活。

    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        bias (bool): 是否使用偏置，默认为True
        config_str (str): 非线性激活函数配置字符串，默认为"batchnorm-relu"
    """
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)  # 获取非线性激活函数
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)  # 1D卷积

    def forward(self, x):
        """前向传播函数，先应用非线性激活再进行卷积操作

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 经非线性激活和卷积后的输出特征
        """
        x = self.nonlinear(x)  # 应用非线性激活
        x = self.linear(x)  # 1D卷积操作
        return x


class DenseLayer(torch.nn.Module):
    """DenseLayer类：一个简单的密集卷积层，应用非线性激活。

    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        bias (bool): 是否使用偏置，默认为False
        config_str (str): 非线性激活函数配置字符串，默认为"batchnorm-relu"
    """
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super(DenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)  # 1D卷积
        self.nonlinear = get_nonlinear(config_str, out_channels)  # 获取非线性激活函数

    def forward(self, x):
        """前向传播函数，先进行卷积操作，再应用非线性激活

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 经卷积和非线性激活后的输出
        """
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)  # 处理2D输入
        else:
            x = self.linear(x)  # 处理其他输入
        x = self.nonlinear(x)  # 应用非线性激活
        return x


class CAMPPlus(torch.nn.Module):
    """CAMPPlus类：基于深度卷积网络和通道注意力机制（CAM）的音频特征提取网络

    参数：
        feat_dim (int): 输入特征的维度，默认为80
        embedding_size (int): 输出的嵌入向量维度，默认为192
        growth_rate (int): 特征图的增长率，默认为32
        bn_size (int): 批归一化层的规模，默认为4
        init_channels (int): 初始通道数，默认为128
        config_str (str): 非线性激活函数配置字符串，默认为"batchnorm-relu"
        memory_efficient (bool): 是否启用内存高效操作，默认为True
        output_level (str): 输出级别，可选"segment"或"frame"，默认为"segment"
    """
    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)  # 特征提取头
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        # 堆叠多个CAMDenseTDNNBlock层
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        # 根据输出级别选择输出方式
        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())  # 统计池化
            self.xvector.add_module(
                "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
            )
        else:
            assert (
                self.output_level == "frame"
            ), "`output_level` should be set to 'segment' or 'frame'. "

        # 权重初始化
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播函数

        参数：
            x (Tensor): 输入特征张量

        返回：
            Tensor: 输出特征张量
        """
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)  # 特征提取头
        x = self.xvector(x)  # xvector网络
        if self.output_level == "frame":
            x = x.transpose(1, 2)  # 调整输出形状
        return x

    def inference(self, audio_list):
        """推理函数，用于从音频列表中提取特征并进行推理

        参数：
            audio_list (List): 输入的音频列表

        返回：
            Tensor: 推理结果
        """
        speech, speech_lengths, speech_times = extract_feature(audio_list)
        results = self.forward(speech.to(torch.float32))  # 前向传播
        return results
