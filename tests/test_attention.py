import torch

from src.models import TransformerLanguageModel
from src.models.attention import MultiHeadCausalSelfAttention


def test_causal_mask_is_lower_triangular():
    attention = MultiHeadCausalSelfAttention(16, 4, 0.0, 8)
    assert torch.equal(attention.causal_mask, torch.tril(torch.ones(8, 8, dtype=torch.bool)))


def test_future_tokens_cannot_change_past_logits():
    torch.manual_seed(7)
    model = TransformerLanguageModel(20, 8, 16, 4, 2, 0.0).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    b = a.clone()
    b[:, 5:] = torch.tensor([15, 16, 17])
    logits_a, _, _ = model(a)
    logits_b, _, _ = model(b)
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-6)


def test_attention_future_weights_are_zero():
    model = TransformerLanguageModel(12, 6, 12, 3, 1, 0.0).eval()
    _, _, attentions = model(torch.arange(6).unsqueeze(0), return_attention=True)
    future = torch.triu(attentions[0][0, 0], diagonal=1)
    assert torch.count_nonzero(future) == 0

