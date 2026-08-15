import torch
from torch.utils.data import Dataset


def contiguous_split(text, train_ratio=0.9, val_ratio=0.05):
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must leave a non-empty test split")
    train_end = int(len(text) * train_ratio)
    val_end = train_end + int(len(text) * val_ratio)
    return {"train": text[:train_end], "val": text[train_end:val_end], "test": text[val_end:]}


class AutoregressiveDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        if len(self.tokens) <= block_size:
            raise ValueError("Split must contain more tokens than block_size")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        x = self.tokens[index : index + self.block_size]
        y = self.tokens[index + 1 : index + self.block_size + 1]
        return x, y

