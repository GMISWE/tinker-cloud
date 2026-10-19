"""Model shapes the miles builder tests are measured on."""

# Qwen2.5-0.5B, the shape every number in the pack-length investigation was measured on.
QWEN25_05B = {
    "num_layers": 24,
    "hidden_size": 896,
    "ffn_hidden_size": 4864,
    "num_attention_heads": 14,
    "num_query_groups": 2,
    "kv_channels": 64,
    "vocab_size": 151936,
    "norm_epsilon": 1e-6,
    "rotary_base": 1000000,
    "tie_word_embeddings": True,
}
