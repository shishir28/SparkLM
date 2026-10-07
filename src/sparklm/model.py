import torch
from torch import nn

from sparklm.layers import LayerNorm
from sparklm.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_length = config["context_length"]
        self.tok_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"]
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.context_length, config["embedding_dim"])
        )
        self.drop_embedding = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config["num_layers"])
        )
        self.final_layer_norm = LayerNorm(config["embedding_dim"])
        self.out_head = nn.Linear(
            config["embedding_dim"], config["vocab_size"], bias=False
        )

    def forward(self, token_ids):
        _, seq_length = token_ids.shape
        if seq_length > self.context_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds context length "
                f"{self.context_length}"
            )

        x = self.tok_embedding(token_ids)
        x = x + self.pos_embedding[:, :seq_length]
        x = self.drop_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        return self.out_head(x)
