import torch


def validate_sampling(temperature=1.0, top_k=None, top_p=1.0):
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def sample_logits(logits, temperature=1.0, top_k=None, top_p=1.0, generator=None):
    validate_sampling(temperature, top_k, top_p)
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1:
        sorted_logits, indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, sorted_logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)


def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=1.0, seed=42):
    validate_sampling(temperature, top_k, top_p)
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[:, -getattr(model, "block_size", tokens.size(1)) :]
            logits, _, _ = model(context)
            next_id = sample_logits(logits[:, -1], temperature, top_k, top_p, generator)
            tokens = torch.cat((tokens, next_id), dim=1)
    return tokens[0].tolist()

