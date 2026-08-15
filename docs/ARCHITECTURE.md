# Architecture

The model is a decoder-only Pre-LayerNorm Transformer implemented from `Linear`, `Embedding`, `LayerNorm`, `GELU` and tensor operations. Q/K/V are projected jointly, reshaped into heads, scaled by `sqrt(head_dim)`, masked by a registered lower-triangular buffer, normalized by softmax and combined with values. Every block has attention and feed-forward residual paths. `return_attention=True` exposes post-softmax weights for diagnostics without changing the default training path.

Text is split contiguously before `AutoregressiveDataset` creates shifted windows. Therefore no window can cross a train/validation/test boundary.

