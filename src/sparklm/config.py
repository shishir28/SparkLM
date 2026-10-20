
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "embedding_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT2_MODEL_CONFIGS = {
 "gpt2-small (124M)": {
    "embedding_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
 },
    "gpt2-medium (355M)": {
        "embedding_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
    },
    "gpt2-large (774M)": {
        "embedding_dim": 1280,
        "num_layers": 36,
        "num_heads": 20,
    },
    "gpt2-xl (1558M)": {
        "embedding_dim": 1600,
        "num_layers": 48,
        "num_heads": 25,
    },
}
