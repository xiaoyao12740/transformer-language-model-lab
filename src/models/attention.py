import math

import torch
from torch import nn


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads, dropout, block_size):
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x, return_attention=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda value: value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = self.attn_dropout(weights) @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        output = self.resid_dropout(self.projection(attended))
        return (output, weights) if return_attention else (output, None)

