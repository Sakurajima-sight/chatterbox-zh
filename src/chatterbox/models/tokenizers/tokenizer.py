import logging

import torch
from tokenizers import Tokenizer


# 特殊标记
SOT = "[START]"  # 开始标记
EOT = "[STOP]"   # 结束标记
UNK = "[UNK]"    # 未知标记
SPACE = "[SPACE]"  # 空格标记
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class ZhTokenizer:
    """
    用于处理中文文本的分词器类。
    该类提供了加载词汇表、检查特殊标记、文本编码和解码功能。
    """

    def __init__(self, vocab_file_path):
        """
        初始化分词器并加载词汇表文件。

        参数:
        vocab_file_path (str): 词汇表文件的路径，通常是一个包含词汇和相关配置的JSON文件。
        """
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)  # 加载词汇表文件
        self.check_vocabset_sot_eot()  # 检查词汇表是否包含特殊标记 [START] 和 [STOP]

    def check_vocabset_sot_eot(self):
        """
        检查词汇表中是否包含 [START] 和 [STOP] 特殊标记。

        如果词汇表缺少这些标记，会抛出错误。
        """
        voc = self.tokenizer.get_vocab()  # 获取词汇表
        assert SOT in voc, f"{SOT} not found in vocab"  # 确保词汇表包含 [START] 标记
        assert EOT in voc, f"{EOT} not found in vocab"  # 确保词汇表包含 [STOP] 标记

    def text_to_tokens(self, text: str):
        """
        将输入的文本转换为 token（数字表示的分词）。

        参数:
        text (str): 输入的文本，通常是一个中文句子。

        返回:
        torch.IntTensor: 转换后的 token 张量，形状为 [1, N]，N 是 token 的数量。
        """
        text_tokens = self.encode(text)  # 使用 encode 方法对文本进行编码
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)  # 将编码后的 token 转换为 Tensor，并增加一个 batch 维度
        return text_tokens

    def encode(self, txt: str, verbose=False):
        """
        对文本进行编码：首先清理文本（替换空格），然后使用分词器进行编码。

        参数:
        txt (str): 输入文本，通常是中文文本。
        verbose (bool): 是否打印详细信息，默认不打印。

        返回:
        list: 编码后的 token ID 列表，每个 token 用对应的 ID 表示。
        """
        txt = txt.replace(' ', SPACE)  # 将空格替换为 [SPACE] 标记，确保空格被当做特殊标记处理
        code = self.tokenizer.encode(txt)  # 使用分词器对文本进行编码
        ids = code.ids  # 获取编码后的 token IDs
        return ids

    def decode(self, seq):
        """
        将 token ID 序列解码回文本，保留所有特殊标记。

        参数:
        seq (torch.Tensor or list): 输入的 token ID 序列，可以是 tensor 或 list 格式。

        返回:
        str: 解码后的文本，可能包含特殊标记。
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()  # 如果输入是 tensor，将其转换为 numpy 数组

        txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)  # 解码并保留特殊标记
        txt = txt.replace(' ', '')  # 移除多余的空格
        txt = txt.replace(SPACE, ' ')  # 将 [SPACE] 转换为普通空格
        txt = txt.replace(EOT, '')  # 移除 [STOP] 标记
        txt = txt.replace(UNK, '')  # 移除 [UNK] 标记
        return txt
