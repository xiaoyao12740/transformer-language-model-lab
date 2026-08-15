from torch import nn

from src.models.attention import MultiHeadCausalSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, heads, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(embedding_dim, heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        attended, weights = self.attention(self.ln1(x), return_attention)
        x = x + attended
        x = x + self.ffn(self.ln2(x))
        return x, weights

