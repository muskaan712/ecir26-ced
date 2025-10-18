#!/usr/bin/env python3
# encoder_eval_all5.py
#
# Train & evaluate 5 encoder-only models on SynCED EN-DE dataset.

import os
import re
import json
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# ─── Config ──────────────────────────────────────────────────────────────
TRAIN_FILE = "/your_path_to/the.tsv"
DEV_FILE   = "/your_path_to/the.tsv"
OUTPUT_ROOT = "output_all5_models"
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_RATIO = 0.1
OPT_BETAS = (0.9, 0.999)
OPT_EPS = 1e-6

MODEL_CONFIGS = {
    "bert-base-uncased": {
        "hf_name": "google/bert_uncased_L-12_H-768_A-12",
        "batch_size": 32,
        "epochs": 3,
        "lr": 2e-5
    },
    "ModernBERT-base": {
        "hf_name": "answerdotai/ModernBERT-base",
        "batch_size": 32,
        "epochs": 3,
        "lr": 2e-5
    },
    "ModernBERT-large": {
        "hf_name": "answerdotai/ModernBERT-large",
        "batch_size": 16,
        "epochs": 3,
        "lr": 2e-5
    },
    "xlm-roberta-large": {
        "hf_name": "xlm-roberta-large",
        "batch_size": 16,
        "epochs": 3,
        "lr": 5e-6
    },
    "mmBERT-base": {
        "hf_name": "jhu-clsp/mmBERT-base",
        "batch_size": 32,
        "epochs": 3,
        "lr": 2e-5
    }
}

# ─── Dataset ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", str(text))
    return text.lower()

class CEDDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_length=256):
        df = pd.read_csv(tsv_path, sep="\t", header=None)
        if df.shape[1] == 3:
            df.columns = ['src','mt','label']
        elif df.shape[1] == 5:
            df.columns = ['id','src','mt','toklabels','label']
        else:
            raise ValueError(f"Unexpected TSV format with {df.shape[1]} columns: {tsv_path}")

        self.src = df['src'].astype(str).tolist()
        self.mt = df['mt'].astype(str).tolist()
        self.labels = df['label'].map({'ERR':1, 'NOT':0}).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        src_text = clean_text(self.src[idx])
        mt_text = clean_text(self.mt[idx])
        enc = self.tokenizer(src_text, mt_text,
                             truncation=True,
                             padding="max_length",
                             max_length=self.max_length,
                             return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['src'] = src_text
        item['mt'] = mt_text
        return item

# ─── Model ────────────────────────────────────────────────────────────────
class CEDModel(nn.Module):
    def __init__(self, hf_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2)   # raw logits only
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h0 = out.last_hidden_state[:,0,:]
        return self.classifier(h0)

# ─── Evaluation ───────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            logits = model(ids, mask)
            pred = torch.argmax(logits, -1)
            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())
    mcc = matthews_corrcoef(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, labels=[1,0], zero_division=0)
    cm = confusion_matrix(trues, preds, labels=[1,0])
    return {'mcc': mcc, 'f1_err': f1[0], 'f1_not': f1[1],
            'precision_err': prec[0], 'recall_err': rec[0],
            'confusion_matrix': cm.tolist()}

# ─── Training/Eval Routine ────────────────────────────────────────────────
def run_model(model_key, cfg):
    print(f"\n=== Running {model_key} ===")
    out_dir = os.path.join(OUTPUT_ROOT, model_key.replace("/","_"))
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    train_ds = CEDDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    dev_ds   = CEDDataset(DEV_FILE, tokenizer, MAX_LENGTH)

    labels = train_ds.labels
    counts = [labels.count(0), labels.count(1)]
    total = sum(counts)
    weights = [total/(2*c) for c in counts]
    sample_weights = [weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler)
    dev_loader   = DataLoader(dev_ds, batch_size=cfg["batch_size"])

    model = CEDModel(cfg["hf_name"]).to(DEVICE)
    opt = AdamW(model.parameters(), lr=cfg["lr"], betas=OPT_BETAS, eps=OPT_EPS)
    total_steps = len(train_loader)*cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO*total_steps), total_steps)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, cfg["epochs"]+1):
        model.train(); ep_loss = 0
        for batch in tqdm(train_loader, desc=f"{model_key} Epoch {ep}"):
            opt.zero_grad()
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            logits = model(ids, mask)
            loss = criterion(logits, y)
            loss.backward(); opt.step(); sched.step()
            ep_loss += loss.item()
        print(f"Epoch {ep} - avg loss: {ep_loss/len(train_loader):.4f}")
        dev_metrics = evaluate(model, dev_loader)
        print(f"Dev MCC: {dev_metrics['mcc']:.4f}, "
              f"F1-ERR: {dev_metrics['f1_err']:.4f}, "
              f"F1-NOT: {dev_metrics['f1_not']:.4f}")

    test_metrics = evaluate(model, dev_loader)
    print(f"Test MCC: {test_metrics['mcc']:.4f}, "
          f"F1-ERR: {test_metrics['f1_err']:.4f}, "
          f"F1-NOT: {test_metrics['f1_not']:.4f}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"dev": dev_metrics, "test": test_metrics}, f, indent=2)

    with open(os.path.join(out_dir, "predictions.tsv"), "w", encoding="utf-8") as f:
        f.write("src\tmt\ttrue\tpred\n")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Saving preds"):
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                y = batch['label'].tolist()
                logits = model(ids, mask)
                preds = torch.argmax(logits, -1).tolist()
                for src, mt, t, p in zip(batch['src'], batch['mt'], y, preds):
                    f.write(f"{src}\t{mt}\t{t}\t{p}\n")

# ─── Entrypoint ──────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for model_key, cfg in MODEL_CONFIGS.items():
        run_model(model_key, cfg)

if __name__ == "__main__":
    main()
