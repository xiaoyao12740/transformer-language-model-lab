import torch

from src.models import BigramLanguageModel, TransformerLanguageModel


def test_forward_shape_and_finite_loss():
    x = torch.randint(0, 15, (3, 8))
    y = torch.randint(0, 15, (3, 8))
    for model in (BigramLanguageModel(15), TransformerLanguageModel(15, 8, 16, 4, 2, 0.0)):
        logits, loss, _ = model(x, y)
        assert logits.shape == (3, 8, 15)
        assert torch.isfinite(loss)


def test_tiny_batch_overfit():
    torch.manual_seed(1)
    model = TransformerLanguageModel(8, 4, 16, 4, 1, 0.0)
    x = torch.tensor([[0, 1, 2, 3]] * 4)
    y = torch.tensor([[1, 2, 3, 4]] * 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = model(x, y)[1].item()
    for _ in range(25):
        loss = model(x, y)[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.25

