# 定义一个 Llama 模型的配置字典，适用于 520M 的模型版本
LLAMA_520M_CONFIG_DICT = dict(
    # 一个小的任意数字，不会在加载时引起问题
    # 由于自定义输入层，以下参数在模型加载时并不会使用
    vocab_size=8,  # 词汇表大小（虽然实际在此模型中不使用）
    
    # 加载大多数预训练 1B 权重时需要的默认参数
    max_position_embeddings=131072,  # 最大位置嵌入（表示最大支持的序列长度）
    hidden_size=1024,  # 隐藏层的维度大小
    intermediate_size=4096,  # 中间层大小，通常用于全连接层的输出大小
    num_hidden_layers=30,  # 隐藏层的数量
    num_attention_heads=16,  # 注意力头的数量
    attn_implementation="sdpa",  # 使用的注意力机制实现方式（这里是 sdpa）
    head_dim=64,  # 每个注意力头的维度
    tie_word_embeddings=False,  # 是否将词嵌入与输出层的权重共享
    hidden_act="silu",  # 激活函数，这里使用的是 SiLU（Sigmoid Linear Unit）
    attention_bias=False,  # 是否使用注意力偏置
    attention_dropout=0.0,  # 注意力机制中的 dropout 比率
    initializer_range=0.02,  # 权重初始化范围
    mlp_bias=False,  # 是否为 MLP 层使用偏置
    model_type="llama",  # 模型类型，这里指定为 "llama"
    num_key_value_heads=16,  # 在 Transformer 中的键值对头数
    pretraining_tp=1,  # 预训练过程中的参数（通常是通过并行化提升效率）
    rms_norm_eps=1e-05,  # RMSNorm 的 epsilon 值，用于数值稳定性
    rope_scaling=dict(
        factor=8.0,  # 缩放因子
        high_freq_factor=4.0,  # 高频缩放因子
        low_freq_factor=1.0,  # 低频缩放因子
        original_max_position_embeddings=8192,  # 原始最大位置嵌入大小
        rope_type="llama3"  # ROPE（相对位置编码）的类型，这里使用 llama3 类型
    ),
    rope_theta=500000.0,  # ROPE 使用的 theta 参数
    torch_dtype="bfloat16",  # 模型使用的精度（bfloat16 是 TensorFlow 中常用的低精度）
    use_cache=True,  # 是否使用缓存
)

# 将 Llama_520M 的配置字典添加到模型配置的字典中
LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG_DICT,  # Llama_520M 配置
}
