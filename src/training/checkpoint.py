from pathlib import Path

import torch

from src.models import BigramLanguageModel, TransformerLanguageModel
from src.tokenizer import CharacterTokenizer


def save_checkpoint(path, model, optimizer, tokenizer, metadata):
    payload = {"model_type": "bigram" if isinstance(model, BigramLanguageModel) else "transformer", "model_config": model.config, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer else None, "tokenizer": tokenizer.to_dict(), **metadata}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    cls = BigramLanguageModel if payload["model_type"] == "bigram" else TransformerLanguageModel
    model = cls(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, CharacterTokenizer.from_dict(payload["tokenizer"]), payload

