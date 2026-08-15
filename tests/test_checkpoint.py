import torch

from src.models import TransformerLanguageModel
from src.tokenizer import CharacterTokenizer
from src.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    tokenizer = CharacterTokenizer().fit("abc")
    model = TransformerLanguageModel(3, 4, 8, 2, 1, 0.0)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, None, tokenizer, {"step": 2, "train_loss": 1.0, "validation_loss": 1.1, "seed": 42, "corpus_sha256": "abc"})
    loaded, loaded_tokenizer, payload = load_checkpoint(path)
    x = torch.tensor([[0, 1, 2]])
    assert torch.allclose(model(x)[0], loaded(x)[0])
    assert loaded_tokenizer.decode([0, 1, 2]) == "abc"
    assert payload["step"] == 2

