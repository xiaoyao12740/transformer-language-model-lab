import math

import torch


class CountBigram:
    def __init__(self, vocab_size, alpha):
        self.vocab_size, self.alpha = vocab_size, alpha

    def fit(self, tokens):
        counts = torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.float64)
        for current, target in zip(tokens[:-1], tokens[1:], strict=True):
            counts[current, target] += 1
        self.log_probs = ((counts + self.alpha) / (counts.sum(1, keepdim=True) + self.alpha*self.vocab_size)).log()
        return self

    def evaluate(self, tokens, positions):
        targets=torch.tensor([tokens[p] for p in positions]); current=torch.tensor([tokens[p-1] for p in positions])
        logits=self.log_probs[current]; nll=-logits[torch.arange(len(targets)),targets].sum().item(); ce=nll/len(targets)
        return {"cross_entropy":ce,"perplexity":math.exp(ce),"bits_per_character":ce/math.log(2),"accuracy":float((logits.argmax(1)==targets).double().mean())}


def select_alpha(train_tokens, val_tokens, positions, alphas):
    scored=[(a,CountBigram(max(train_tokens)+1,a).fit(train_tokens)) for a in alphas]
    return min(scored,key=lambda item:item[1].evaluate(val_tokens,positions)["cross_entropy"])
