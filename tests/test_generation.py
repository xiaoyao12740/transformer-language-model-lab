import pytest
import torch

from src.generation.sampling import generate, sample_logits, validate_sampling
from src.models import TransformerLanguageModel


def test_generation_length_and_seed_reproducibility():
    model = TransformerLanguageModel(10, 8, 16, 4, 1, 0.0)
    a = generate(model, [1, 2], 7, 0.8, 5, 0.9, 12)
    b = generate(model, [1, 2], 7, 0.8, 5, 0.9, 12)
    assert len(a) == 9
    assert a == b


@pytest.mark.parametrize("values", [(-1, None, 1), (1, 0, 1), (1, None, 0), (1, None, 1.1)])
def test_sampling_argument_validation(values):
    with pytest.raises(ValueError):
        validate_sampling(*values)


def test_top_k_sampling():
    logits = torch.tensor([[0.0, 10.0, 9.0, 8.0]])
    results = {sample_logits(logits, top_k=2).item() for _ in range(20)}
    assert results <= {1, 2}

