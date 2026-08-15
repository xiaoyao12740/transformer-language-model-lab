import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch

from src.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports/figures"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics_dir = ROOT / "reports/metrics"
    tables_dir = ROOT / "reports/tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads((ROOT / f"checkpoints/{name}/metrics.json").read_text()) for name in ("bigram", "transformer")]
    formal = {"experiment": "formal-cpu-seed-42", "models": {row["model"]: row for row in rows}}
    (metrics_dir / "formal_metrics.json").write_text(json.dumps(formal, indent=2), encoding="utf-8")
    fields = ["model", "test_cross_entropy", "test_perplexity", "test_bits_per_character", "test_accuracy", "parameter_count", "checkpoint_size_bytes", "training_time_seconds", "tokens_per_second"]
    with (tables_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in rows)

    fig, ax = plt.subplots(figsize=(11, 4)); ax.axis("off")
    ax.text(0.5, 0.85, "Character IDs + positions", ha="center", bbox=dict(boxstyle="round", fc="#dbeafe"), fontsize=13)
    ax.text(0.5, 0.55, "3 × Pre-LN decoder block\ncausal MHA → residual → FFN → residual", ha="center", bbox=dict(boxstyle="round", fc="#dcfce7"), fontsize=13)
    ax.text(0.5, 0.2, "Final LayerNorm → 64-character logits", ha="center", bbox=dict(boxstyle="round", fc="#fef3c7"), fontsize=13)
    for y1, y2 in ((0.78, 0.65), (0.45, 0.3)): ax.annotate("", (0.5, y2), (0.5, y1), arrowprops=dict(arrowstyle="->", lw=2))
    save(fig, "01_system_architecture.png")

    mask = torch.tril(torch.ones(16, 16)).numpy(); fig, ax = plt.subplots(figsize=(6, 5)); im=ax.imshow(mask, cmap="Blues"); ax.set_title("Causal attention mask: position i sees only ≤ i"); ax.set_xlabel("Key position"); ax.set_ylabel("Query position"); fig.colorbar(im, ax=ax); save(fig, "02_causal_attention_mask.png")

    tr = formal["models"]["transformer"]; history=tr["history"]
    fig, ax=plt.subplots(figsize=(7,4)); ax.plot([x["step"] for x in history],[x["train_loss"] for x in history],marker="o",label="train batch"); ax.plot([x["step"] for x in history],[x["val_loss"] for x in history],marker="o",label="validation"); ax.set(xlabel="Step",ylabel="Cross entropy",title="Formal Transformer learning curve"); ax.legend(); save(fig,"03_train_validation_loss.png")

    fig, axes=plt.subplots(1,2,figsize=(9,4)); names=[r["model"] for r in rows]; axes[0].bar(names,[r["test_perplexity"] for r in rows],color=["#94a3b8","#2563eb"]); axes[0].set_title("Character perplexity ↓"); axes[1].bar(names,[r["test_bits_per_character"] for r in rows],color=["#94a3b8","#16a34a"]); axes[1].set_title("Bits per character ↓"); save(fig,"04_perplexity_bpc.png")

    fig, axes=plt.subplots(1,2,figsize=(9,4)); axes[0].bar(names,[r["test_cross_entropy"] for r in rows]); axes[0].set_title("Test cross entropy ↓"); axes[1].bar(names,[r["test_accuracy"] for r in rows]); axes[1].set_title("Next-character accuracy ↑"); save(fig,"05_baseline_comparison.png")

    model, tokenizer, _=load_checkpoint(ROOT/"checkpoints/transformer/best.pt"); prompt="To be or not to be"; x=torch.tensor([tokenizer.encode(prompt)]); model.eval()
    with torch.no_grad(): _,_,att=model(x,return_attention=True)
    matrix=att[-1][0,0].numpy(); fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(matrix,cmap="magma"); ax.set_xticks(range(len(prompt)),list(prompt),fontsize=7); ax.set_yticks(range(len(prompt)),list(prompt),fontsize=7); ax.set_title("Real checkpoint attention: final layer, head 0"); save(fig,"06_attention_heatmap.png")

    samples=json.loads((ROOT/"checkpoints/transformer/samples.json").read_text()); fig,ax=plt.subplots(figsize=(12,6)); ax.axis("off"); y=0.95
    for name,text in samples.items(): ax.text(0.01,y,name.replace("_"," "),weight="bold",fontsize=11); ax.text(0.01,y-0.06,text[:210].replace("\n"," ↵ "),fontsize=9,wrap=True); y-=0.3
    ax.set_title("Sampling comparison from the measured checkpoint"); save(fig,"07_sampling_comparison.png")
    print(json.dumps(formal, indent=2))


if __name__ == "__main__": main()
