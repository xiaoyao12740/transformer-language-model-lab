# Transformer Language Model Lab

[简体中文](README_zh-CN.md)

[![CI](https://github.com/xiaoyao12740/transformer-language-model-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaoyao12740/transformer-language-model-lab/actions/workflows/ci.yml)

A from-scratch causal Transformer language-model lab covering character tokenization, masked multi-head self-attention, decoder blocks, leakage-safe corpus splits, reproducible training, held-out evaluation, controlled generation and attention visualization.

![System architecture](reports/figures/01_system_architecture.png)

## Measured results

One CPU formal run, seed 42, on a contiguous 600,000-character public-domain Shakespeare excerpt. Both models use the same tokenizer and train/validation/test boundaries; the test split is evaluated only after selecting the best validation checkpoint.

| Model | Test CE ↓ | Character PPL ↓ | BPC ↓ | Next-char accuracy ↑ | Parameters | Train time |
|---|---:|---:|---:|---:|---:|---:|
| Bigram | 3.9272 | 50.7644 | 5.6657 | 6.20% | 4,096 | 0.83 s |
| Causal Transformer | **2.1271** | **8.3907** | **3.0688** | **38.01%** | 357,280 | 78.03 s |

Character-level perplexity is tokenizer-dependent and must not be compared directly with subword perplexity reported for GPT, LLaMA or other models.

![Learning curve](reports/figures/03_train_validation_loss.png)
![Baseline comparison](reports/figures/05_baseline_comparison.png)

## Why this project

The original teaching prototype introduced embeddings and `TransformerEncoder`, but next-character labels were shifted while attention remained bidirectional. A position could therefore inspect future input tokens that contained its target. This lab converts that lesson into a decoder-only autoregressive implementation and machine-checks the causal invariant.

## Why causal masking matters

For sequence position `t`, attention scores to keys `> t` are set to negative infinity before softmax. Training and generation therefore share the same information boundary.

![Causal mask](reports/figures/02_causal_attention_mask.png)

`test_future_tokens_cannot_change_past_logits` creates two sequences with identical prefixes and different suffixes, then asserts that every prefix logit is equal within numerical tolerance. A second test checks that all upper-triangular attention weights are exactly zero.

## Architecture

`CharacterTokenizer → learned token + position embeddings → 3 × Pre-LN decoder block → final LayerNorm → LM head`

Each decoder block implements Q/K/V projections, scaled dot-product scores, lower-triangular masking, softmax, multiple heads, projection, residual paths and a GELU feed-forward network using basic PyTorch modules—not `nn.TransformerEncoderLayer`.

## Dataset and provenance

- Source: [char-rnn tiny Shakespeare transcription](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
- Underlying works: William Shakespeare, public domain
- Normalization: LF line endings; first 600,000 characters
- SHA-256: `2020ddbb2988648b625422110087228b5cddf29572655b5cc56d3f1543d436f8`
- Vocabulary: 64 characters
- Split: contiguous 90% train / 5% validation / 5% test, **before** sliding windows

This prevents near-identical overlapping windows from crossing split boundaries. `data/tiny_corpus.txt` is an original smoke-test corpus and is not the formal experiment.

## Training, checkpoint and evaluation

The small configuration uses block size 96, width 96, four heads, three layers, AdamW, mini-batches, gradient clipping, periodic validation and best/last checkpoints. A checkpoint contains model and optimizer states, model config, tokenizer vocabulary, step, losses, seed and corpus hash, so a new process can load and generate without reconstructing hidden state.

Formal artifacts: [metrics](reports/metrics/formal_metrics.json), [comparison table](reports/tables/model_comparison.csv), [experiment protocol](docs/EXPERIMENTS.md), [model card](docs/MODEL_CARD.md).

## Generation and attention explorer

Generation supports greedy (`temperature=0`), temperature, top-k and nucleus top-p sampling with validation and seeded reproducibility.

![Sampling comparison](reports/figures/07_sampling_comparison.png)

The bilingual Streamlit app provides Generate, next-character probabilities, selectable layer/head attention and an experiment dashboard. The heatmap below is read from the committed measured checkpoint—not a mockup.

![Measured attention](reports/figures/06_attention_heatmap.png)

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"  # Windows
python -m scripts.prepare_data
python -m scripts.experiment --model transformer --config configs/small.yaml --corpus data/formal_corpus.txt --output checkpoints
python -m scripts.build_reports
streamlit run app/streamlit_app.py
```

## Tests and CI

```bash
ruff check .
pytest -q
```

The suite covers tokenizer roundtrip, shift alignment, contiguous splits, causal mask, future-token invariance, forward/loss, tiny overfit, checkpoint reload, sampling validation and deterministic generation. CI runs Python 3.10/3.11/3.12 plus a tiny train → save → reload → generate smoke chain. **CI smoke run is not the formal language-model experiment.**

## Limitations

This is a small character model trained once on CPU and is not a general-purpose LLM. It has no subword tokenizer, broad factual knowledge, instruction tuning, safety alignment or production serving guarantees. A single seed is not mean ± standard deviation, generated Shakespeare-like text may be incoherent, and attention weights are diagnostics rather than explanations of intent.

