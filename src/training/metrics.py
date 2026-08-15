import math

import torch


def derived_metrics(loss):
    return {"cross_entropy": loss, "perplexity": math.exp(min(loss, 20)), "bits_per_character": loss / math.log(2)}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    losses, correct, count = [], 0, 0
    for index, (x, y) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, loss, _ = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    loss = sum(losses) / len(losses)
    return {**derived_metrics(loss), "accuracy": correct / count}

