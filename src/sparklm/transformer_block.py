from torch import nn

from sparklm.attention.multi_head import MultiHeadAttentionCombined
from sparklm.layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embedding_dim = cfg["embedding_dim"]
        self.att = MultiHeadAttentionCombined(
            d_in=embedding_dim,
            d_out=embedding_dim,
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"],
            qkv_bias=cfg.get("qkv_bias", False),
        )
        self.ff = FeedForward({"emb_dim": embedding_dim})
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut
