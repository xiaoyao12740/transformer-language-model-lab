import random
import time

import torch

from src.training.checkpoint import save_checkpoint
from src.training.metrics import evaluate


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def train(model, train_loader, val_loader, tokenizer, config, checkpoint_dir, corpus_sha256, device="cpu"):
    cfg = config["training"]
    seed_everything(cfg["seed"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    iterator = iter(train_loader)
    history, best = [], float("inf")
    started = time.perf_counter()
    tokens_seen = 0
    for step in range(1, cfg["max_steps"] + 1):
        model.train()
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        optimizer.step()
        tokens_seen += y.numel()
        if step == 1 or step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
            val = evaluate(model, val_loader, device, cfg["eval_batches"])
            row = {"step": step, "train_loss": loss.item(), "val_loss": val["cross_entropy"]}
            history.append(row)
            metadata = {"step": step, "train_loss": loss.item(), "validation_loss": val["cross_entropy"], "seed": cfg["seed"], "corpus_sha256": corpus_sha256}
            save_checkpoint(f"{checkpoint_dir}/last.pt", model, optimizer, tokenizer, metadata)
            if val["cross_entropy"] < best:
                best = val["cross_entropy"]
                save_checkpoint(f"{checkpoint_dir}/best.pt", model, optimizer, tokenizer, metadata)
    elapsed = time.perf_counter() - started
    return history, {"training_time_seconds": elapsed, "tokens_per_second": tokens_seen / elapsed, "best_validation_loss": best}

