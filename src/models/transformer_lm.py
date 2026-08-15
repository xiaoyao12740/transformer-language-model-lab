import torch
from torch import nn
from torch.nn import functional as F

from src.models.block import DecoderBlock


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, heads, layers, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, block_size=block_size, embedding_dim=embedding_dim, heads=heads, layers=layers, dropout=dropout)
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList([DecoderBlock(embedding_dim, heads, dropout, block_size) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens, targets=None, return_attention=False):
        _, length = tokens.shape
        if length > self.block_size:
            raise ValueError("Sequence exceeds block_size")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attentions = []
        for block in self.blocks:
            x, weights = block(x, return_attention)
            if return_attention:
                attentions.append(weights)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, attentions

