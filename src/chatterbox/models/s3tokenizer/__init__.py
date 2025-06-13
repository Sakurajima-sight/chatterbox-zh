from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)

SOS = SPEECH_VOCAB_SIZE  # SoS（序列开始）token的ID
EOS = SPEECH_VOCAB_SIZE + 1  # EOS（序列结束）token的ID

def drop_invalid_tokens(x):
    """
    删除无效的 SoS 和 EOS token

    该函数用于去除序列中的 SoS 和 EOS 标记，确保只有有效的 token 留在序列中。
    函数要求输入的形状为 1D 或者 2D，但 batch size 只能为 1。
    
    参数:
    x: torch.Tensor
        输入的token序列，可能包含 SoS 和 EOS 标记。
        
    返回:
    torch.Tensor
        去除 SoS 和 EOS 标记后的 token 序列。
    """
    # 确保输入是1D或2D且batch size为1
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "目前仅支持 batch size 为 1"

    # 如果包含SoS token，找出其位置并去除SoS之前的token
    if SOS in x:
        s = (x == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
    else:
        s = 0  # 如果没有SoS，直接从第一个token开始

    # 如果包含EOS token，找出其位置并去除EOS之后的token
    if EOS in x:
        e = (x == EOS).nonzero(as_tuple=True)[0].squeeze(0)
    else:
        e = None  # 如果没有EOS，表示序列不包含EOS标记

    # 返回SoS和EOS标记之间的有效token
    x = x[s: e]
    return x
