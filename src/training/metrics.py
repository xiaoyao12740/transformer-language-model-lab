import math

import torch


def derived_metrics(loss):
    return {"cross_entropy": loss, "perplexity": math.exp(min(loss, 20)), "bits_per_character": loss / math.log(2)}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_nll, correct, count = 0.0, 0, 0
    for index, (x, y) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if y.ndim == 1:
            logits, _, _ = model(x)
            logits = logits[:, -1]
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        else:
            logits, _, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += loss.item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    loss = total_nll / count
    return {**derived_metrics(loss), "accuracy": correct / count}
