import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import AutoregressiveDataset, contiguous_split
from src.generation.sampling import generate
from src.models import BigramLanguageModel, TransformerLanguageModel
from src.tokenizer import CharacterTokenizer
from src.training.checkpoint import load_checkpoint
from src.training.metrics import evaluate
from src.training.trainer import train


def run(model_type, config_path, corpus_path, output_dir, device="cpu"):
    config = yaml.safe_load(Path(config_path).read_text())
    text = Path(corpus_path).read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()
    tokenizer = CharacterTokenizer().fit(text)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path / "tokenizer.json")
    splits = contiguous_split(text)
    block = config["model"]["block_size"]
    datasets = {name: AutoregressiveDataset(tokenizer.encode(value), block) for name, value in splits.items()}
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    loaders = {name: DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=name == "train", generator=generator) for name, ds in datasets.items()}
    kwargs = {**config["model"], "vocab_size": tokenizer.vocab_size}
    model = BigramLanguageModel(**kwargs) if model_type == "bigram" else TransformerLanguageModel(**kwargs)
    ckpt_dir = Path(output_dir) / model_type
    history, train_meta = train(model, loaders["train"], loaders["val"], tokenizer, config, ckpt_dir, sha, device)
    model, tokenizer, payload = load_checkpoint(ckpt_dir / "best.pt", device)
    metrics = {"model": model_type, **train_meta, **{f"test_{k}": v for k, v in evaluate(model, loaders["test"], device).items()}, "parameter_count": sum(p.numel() for p in model.parameters()), "checkpoint_size_bytes": (ckpt_dir / "best.pt").stat().st_size, "best_step": payload["step"], "corpus_sha256": sha, "vocab_size": tokenizer.vocab_size, "block_size": block, "history": history}
    prompt = "First Citizen:" if "F" in tokenizer.char_to_id else splits["test"][:8]
    samples = {}
    for label, params in {"greedy": (0, None, 1.0), "temperature_0.8_top_k_20": (0.8, 20, 1.0), "temperature_1.1_top_p_0.9": (1.1, None, 0.9)}.items():
        ids = generate(model, tokenizer.encode(prompt), 160, *params, seed=42)
        samples[label] = tokenizer.decode(ids)
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ckpt_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bigram", "transformer"], required=True)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--corpus", default="data/tiny_corpus.txt")
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.config, args.corpus, args.output), indent=2))


if __name__ == "__main__":
    main()
