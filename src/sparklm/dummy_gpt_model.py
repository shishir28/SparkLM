import torch
import torch.nn as nn
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_embedding= nn.Dropout(cfg["drop_rate"])
        self.trf_block = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_norm = DummyLayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_embedding(in_idx)
        pos_embeds = self.position_embedding(torch.arange(seq_len, device=in_idx.device))
        x = token_embeds + pos_embeds
        x = self.drop_embedding(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

    def forward(self, x):
        return x
