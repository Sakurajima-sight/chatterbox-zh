# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """
    每位置的前馈层。前馈网络作用于序列的每个位置，输出维度与输入维度相同。

    :param idim: 输入维度。
    :param hidden_units: 隐藏单元的数量。
    :param dropout_rate: dropout 的概率。
    :param activation: 激活函数，默认为 ReLU。
    """

    def __init__(
            self,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        """
        构造 PositionwiseFeedForward 对象。

        :param idim: 输入维度。
        :param hidden_units: 隐藏层的单元数。
        :param dropout_rate: dropout 的概率。
        :param activation: 激活函数，默认为 ReLU。
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)  # 第一层线性变换
        self.activation = activation  # 激活函数
        self.dropout = torch.nn.Dropout(dropout_rate)  # dropout 层
        self.w_2 = torch.nn.Linear(hidden_units, idim)  # 第二层线性变换

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        :param xs: 输入张量，形状为 (B, L, D)，其中 B 为批量大小，L 为序列长度，D 为维度
        :return: 输出张量，形状为 (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))  # 依次通过两层前馈网络


class MoEFFNLayer(torch.nn.Module):
    """
    混合专家模型与 Positionwise 前馈网络层结合的实现。
    参考文献：https://arxiv.org/pdf/2305.15663.pdf 中的图 1。
    输出维度与输入维度相同。

    修改自：https://github.com/Lightning-AI/lit-gpt/pull/823
              https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219

    :param n_expert: 专家数量。
    :param n_expert_per_token: 每个token实际使用的专家数量。
    :param idim: 输入维度。
    :param hidden_units: 隐藏单元的数量。
    :param dropout_rate: dropout 的概率。
    :param activation: 激活函数，默认为 ReLU。
    """

    def __init__(
            self,
            n_expert: int,
            n_expert_per_token: int,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)  # 门控机制：根据输入选择专家
        # 创建多个专家，每个专家都是一个 PositionwiseFeedForward
        self.experts = torch.nn.ModuleList(
            PositionwiseFeedForward(idim, hidden_units, dropout_rate,
                                    activation) for _ in range(n_expert))
        self.n_expert_per_token = n_expert_per_token  # 每个token实际使用的专家数量

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        :param xs: 输入张量，形状为 (B, L, D)，其中 B 为批量大小，L 为序列长度，D 为维度。
        :return: 输出张量，形状为 (B, L, D)
        """
        B, L, D = xs.size()  # 获取批量大小、序列长度和输入维度
        xs = xs.view(-1, D)  # 重塑输入张量形状为 (B*L, D)，展开序列
        router = self.gate(xs)  # 获取每个token的专家选择，形状为 (B*L, n_expert)
        logits, indices = torch.topk(
            router, self.n_expert_per_token
        )  # 获取每个token选择的专家的logits和索引，形状为 (B*L, n_expert)
        weights = torch.nn.functional.softmax(
            logits, dim=1,
            dtype=torch.float).to(dtype=xs.dtype)  # 使用 softmax 计算专家的权重，形状为 (B*L, n_expert_per_token)
        
        output = torch.zeros_like(xs)  # 初始化输出张量，形状为 (B*L, D)
        # 为每个专家计算输出，并根据权重加权
        for i, expert in enumerate(self.experts):
            mask = indices == i  # 获取当前专家的选择mask
            batch_idx, ith_expert = torch.where(mask)  # 找到每个token对应的专家
            # 根据专家的权重和输出加权
            output[batch_idx] += weights[batch_idx, ith_expert, None] * expert(
                xs[batch_idx])

        return output.view(B, L, D)  # 将输出张量恢复成 (B, L, D) 的形状
