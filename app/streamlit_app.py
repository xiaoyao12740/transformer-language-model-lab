import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.generation.sampling import generate
from src.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/transformer/best.pt"

st.set_page_config(page_title="Transformer LM Lab", layout="wide")
st.title("Transformer Language Model Lab / 因果语言模型实验室")
st.caption("From-scratch character-level decoder-only Transformer; educational, not a general-purpose LLM.")
if not CHECKPOINT.exists():
    st.warning("Run the formal experiment first / 请先运行正式实验生成 checkpoint。")
    st.stop()
model, tokenizer, payload = load_checkpoint(CHECKPOINT)

tabs = st.tabs(["Generate / 生成", "Next Character", "Attention", "Experiment"])
with tabs[0]:
    prompt = st.text_input("Prompt", "First Citizen:")
    length = st.slider("New characters", 10, 300, 120)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.1)
    top_k = st.slider("Top-k", 1, min(50, tokenizer.vocab_size), 20)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    if st.button("Generate"):
        ids = generate(model, tokenizer.encode(prompt), length, temperature, top_k, top_p, 42)
        st.code(tokenizer.decode(ids))
with tabs[1]:
    prompt = st.text_input("Context", "To be or not to be", key="next")
    x = torch.tensor([tokenizer.encode(prompt)[-model.block_size :]])
    with torch.no_grad():
        probs = torch.softmax(model(x)[0][0, -1], dim=-1)
    values, indexes = torch.topk(probs, 10)
    st.bar_chart({tokenizer.decode([i]): float(v) for v, i in zip(values, indexes)})
with tabs[2]:
    prompt = st.text_input("Attention context", "To be or not to be", key="attention")
    x = torch.tensor([tokenizer.encode(prompt)[-model.block_size :]])
    with torch.no_grad():
        _, _, attention = model(x, return_attention=True)
    layer = st.selectbox("Layer", range(len(attention)))
    head = st.selectbox("Head", range(attention[layer].shape[1]))
    matrix = attention[layer][0, head].numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(matrix, cmap="magma")
    ax.set_xticks(range(len(prompt))); ax.set_xticklabels(list(prompt), fontsize=7)
    ax.set_yticks(range(len(prompt))); ax.set_yticklabels(list(prompt), fontsize=7)
    st.pyplot(fig)
with tabs[3]:
    metrics_path = ROOT / "reports/metrics/formal_metrics.json"
    st.json(json.loads(metrics_path.read_text()) if metrics_path.exists() else payload)

