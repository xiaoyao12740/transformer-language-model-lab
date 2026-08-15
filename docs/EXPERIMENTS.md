# Experiment protocol

Formal run: CPU, PyTorch 2.13, seed 42, first 600,000 normalized Shakespeare characters, 90/5/5 contiguous split. Both Bigram and Transformer use block size 96 and the same character vocabulary. Training runs 500 optimizer steps; validation is checked every 50 steps and chooses `best.pt`; test is evaluated once after reload. The committed JSON/CSV and figures are generated from those checkpoints.

The experiment is intentionally small. CI uses `configs/tiny.yaml` and the original tiny corpus; its metrics must never replace formal metrics.

