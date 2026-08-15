import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import (
    AutoregressiveDataset,
    HeldoutNextTokenDataset,
    contiguous_split,
    evenly_spaced_positions,
)
from src.generation.sampling import generate
from src.models import TransformerLanguageModel
from src.models.count_bigram import select_alpha
from src.tokenizer import CharacterTokenizer
from src.training.checkpoint import load_checkpoint
from src.training.metrics import evaluate
from src.training.trainer import seed_everything, train


def run(model_type, config_path, corpus_path, output_dir, device="cpu"):
    config_bytes=Path(config_path).read_bytes(); config=yaml.safe_load(config_bytes)
    seed_everything(config["training"]["seed"])
    text=Path(corpus_path).read_text(encoding="utf-8"); sha=hashlib.sha256(text.encode()).hexdigest()
    splits=contiguous_split(text); tokenizer=CharacterTokenizer().fit(splits["train"])
    for name in ("val","test"): tokenizer.encode(splits[name])
    output_path=Path(output_dir); output_path.mkdir(parents=True,exist_ok=True); tokenizer.save(output_path/"tokenizer.json")
    encoded={name:tokenizer.encode(value) for name,value in splits.items()}; block=config["model"]["block_size"]
    selection_positions=evenly_spaced_positions(len(encoded["val"]),block,config["training"].get("validation_targets",1024))
    full_val_positions=list(range(block,len(encoded["val"]))); test_positions=list(range(block,len(encoded["test"])))
    common={"corpus_sha256":sha,"config_sha256":hashlib.sha256(config_bytes).hexdigest(),"seed":config["training"]["seed"],"python_version":platform.python_version(),"torch_version":torch.__version__,"train_chars":len(splits["train"]),"validation_chars":len(splits["val"]),"test_chars":len(splits["test"]),"vocabulary_size":tokenizer.vocab_size,"vocabulary_fit_split":"train","heldout_evaluation_protocol":"B-character context; final target scored exactly once","validation_selection_strategy":"evenly spaced deterministic targets across full validation split","validation_selection_target_count":len(selection_positions),"full_validation_target_count":len(full_val_positions),"test_target_count":len(test_positions)}
    if model_type=="bigram":
        alpha,model=select_alpha(encoded["train"],encoded["val"],selection_positions,[0.01,0.1,0.5,1.0])
        metrics={"model":"count_bigram","chosen_alpha":alpha,"validation":model.evaluate(encoded["val"],full_val_positions),**{f"test_{k}":v for k,v in model.evaluate(encoded["test"],test_positions).items()},"parameter_count":tokenizer.vocab_size**2,"checkpoint_size_bytes":0,"training_time_seconds":0.0,**common,"history":[]}
        path=output_path/"bigram"; path.mkdir(exist_ok=True); (path/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); return metrics
    train_ds=AutoregressiveDataset(encoded["train"],block); val_ds=HeldoutNextTokenDataset(encoded["val"],block,selection_positions)
    generator=torch.Generator().manual_seed(config["training"]["seed"]); train_loader=DataLoader(train_ds,batch_size=config["training"]["batch_size"],shuffle=True,generator=generator); val_loader=DataLoader(val_ds,batch_size=config["training"]["batch_size"])
    model=TransformerLanguageModel(vocab_size=tokenizer.vocab_size,**config["model"]); ckpt=output_path/"transformer"
    history,meta=train(model,train_loader,val_loader,tokenizer,config,ckpt,sha,device); model,tokenizer,payload=load_checkpoint(ckpt/"best.pt",device)
    full_val=DataLoader(HeldoutNextTokenDataset(encoded["val"],block),batch_size=64); test=DataLoader(HeldoutNextTokenDataset(encoded["test"],block),batch_size=64)
    metrics={"model":"transformer",**meta,"validation":evaluate(model,full_val,device),**{f"test_{k}":v for k,v in evaluate(model,test,device).items()},"parameter_count":sum(p.numel() for p in model.parameters()),"checkpoint_size_bytes":(ckpt/"best.pt").stat().st_size,"best_step":payload["step"],**common,"history":history}
    samples={}; prompt="First Citizen:" if "F" in tokenizer.char_to_id else splits["test"][:8]
    for label,params in {"greedy":(0,None,1.0),"temperature_0.8_top_k_20":(0.8,20,1.0),"temperature_1.1_top_p_0.9":(1.1,None,0.9)}.items(): samples[label]=tokenizer.decode(generate(model,tokenizer.encode(prompt),160,*params,seed=42))
    (ckpt/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); (ckpt/"samples.json").write_text(json.dumps(samples,indent=2),encoding="utf-8"); return metrics


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=["bigram","transformer"],required=True); p.add_argument("--config",default="configs/tiny.yaml"); p.add_argument("--corpus",default="data/tiny_corpus.txt"); p.add_argument("--output",default="checkpoints"); a=p.parse_args(); print(json.dumps(run(a.model,a.config,a.corpus,a.output),indent=2))
if __name__=="__main__": main()
