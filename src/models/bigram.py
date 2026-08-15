from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, **_):
        super().__init__()
        self.config = {"vocab_size": vocab_size}
        self.transition = nn.Embedding(vocab_size, vocab_size)

    def forward(self, tokens, targets=None, return_attention=False):
        logits = self.transition(tokens)
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, []

