# Experiment protocol

Formal run: CPU, PyTorch 2.13, seed 42, complete 1,115,394-character Shakespeare source, 90/5/5 contiguous split. Tokenizer fits train only. Count Bigram selects alpha 0.5 on validation. Transformer uses 1,024 evenly-spaced validation targets for checkpoint selection. After reload, 55,673 validation and 55,675 test targets are each scored once with a 96-character context and target-weighted NLL.

The experiment is intentionally small. CI uses `configs/tiny.yaml` and the original tiny corpus; its metrics must never replace formal metrics.
