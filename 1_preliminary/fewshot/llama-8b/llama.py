#!/usr/bin/env python3
# fewshot_llama31_8b_batch_generate.py
#
# Critical Error Detection (EN→DE) with Meta-Llama-3.1-8B-Instruct
# - Local cache via snapshot_download (first run only)
# - Llama 3.1 chat template (system + few-shot demos + user)
# - Batched manual generate (fast tokenizer batch path)
# - Forces eager attention (no FlashAttention2 / SDPA)
# - Few-shot demos: 5 ERR + 3 NOT sampled from TRAIN (order: ERR first, then NOT)
#
# pip install "transformers>=4.42.0" accelerate huggingface_hub torch pandas scikit-learn tqdm

import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV    = os.path.join(DATA_DIR, "ende_majority_dev.tsv")
TRAIN_TSV  = os.path.join(DATA_DIR, "ende_majority_train.tsv")

HF_TOKEN   = os.getenv("HF_TOKEN")  # accept license for Meta-Llama-3.1-8B-Instruct on HF
MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_ROOT = "modelcache"
CACHE_DIR  = os.path.join(CACHE_ROOT, MODEL_ID.replace("/", "_"))

# ─── Inference ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 8
MAX_NEW_TOKENS = 3         # only need 'ERR' or 'NOT'
DO_SAMPLE      = False     # deterministic
MIN_NEW_TOKENS = 1         # ensure at least one token

# Attention backend: force eager (no FA2 / SDPA)
ATTN_IMPL = "eager"

# ─── Few-shot config (enable/disable + counts) ─────────────────────────────────
USE_FEW_SHOT        = True
FEW_SHOT_ERR_CNT    = 5
FEW_SHOT_NOT_CNT    = 3
RANDOM_STATE        = 42

# ─── Prompt text ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
    "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
    "or only has minor imperfections.\n"
    "Do not add any explanation, punctuation, or additional text."
)

# ───────────────────────────────────────────────────────────────────────────────

def download_and_cache_model():
    """Download MODEL_ID into CACHE_DIR once, then reuse."""
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=CACHE_DIR,
            token=HF_TOKEN,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.md", "*.py"]
        )

def load_dev(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["id","src","mt","toklabels","label"],
        dtype={"id": str}
    )
    df["label"] = df["label"].str.strip().str.upper()
    df["label_id"] = df["label"].map({"ERR": 1, "NOT": 0})
    return df

def load_train_minimal(path: str) -> pd.DataFrame:
    """Load TRAIN_TSV robustly and keep only src, mt, label (uppercase)."""
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, na_filter=False)
    n = df.shape[1]
    if n >= 5:
        df = df.iloc[:, :5]; df.columns = ["id","src","mt","raw","label"]
    elif n == 4:
        df.columns = ["src","mt","raw","label"]; df.insert(0, "id", range(len(df)))
    elif n == 3:
        df.columns = ["src","mt","label"]; df.insert(0, "id", range(len(df))); df.insert(3,"raw","")
    else:
        raise ValueError(f"Unexpected TRAIN TSV columns: {n}")
    df["label"] = df["label"].str.strip().str.upper()
    return df[["src","mt","label"]]

def sample_few_shot_examples(train_tsv: str,
                             n_err: int,
                             n_not: int,
                             random_state: int = 42):
    """
    Sample 5 ERR (oversample) + 3 NOT (with replacement if needed).
    Return list of {src, mt, label} with ERR first, then NOT.
    """
    train_df = load_train_minimal(train_tsv)
    train_df = train_df[train_df["label"].isin(["ERR","NOT"])]

    def _sample(df, k):
        if len(df) == 0:
            return df
        if len(df) >= k:
            return df.sample(k, random_state=random_state)
        return df.sample(k, replace=True, random_state=random_state)

    err_df = _sample(train_df[train_df["label"] == "ERR"], n_err)
    not_df = _sample(train_df[train_df["label"] == "NOT"], n_not)

    examples = []
    for _, r in err_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "ERR"})
    for _, r in not_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "NOT"})
    return examples

def build_messages_zero_shot(src: str, mt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}\nAnswer with ONLY 'ERR' or 'NOT'."}
    ]

def build_messages_few_shot(examples, src: str, mt: str):
    """
    Few-shot conversational format:
      (system)
      user: EN/DE pair → assistant: gold label
      ...
      user: EN/DE (query)
    ERR examples appear first, then NOT (matches your prior GPT-4o logic).
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user",      "content": f"EN: {ex['src']}\nDE: {ex['mt']}"})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}\nAnswer with ONLY 'ERR' or 'NOT'."})
    return msgs

def extract_label(text: str) -> str:
    """Extract final 'ERR' or 'NOT'. Default to NOT if ambiguous/empty."""
    t = (text or "").upper()
    hits = re.findall(r"\b(ERR|NOT)\b", t)
    return hits[-1] if hits else "NOT"

def main():
    # 1) Cache model locally
    download_and_cache_model()

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(CACHE_DIR, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # best for decoder-only batched gen

    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        CACHE_DIR,
        local_files_only=True,
        device_map="auto",
        torch_dtype=dtype # <- force eager; no FA2/SDPA
    ).eval()

    # 3) Load data
    df = load_dev(DEV_TSV)

    # 4) Build few-shot exemplars ONCE (no leakage into label computation)
    few_shots = None
    if USE_FEW_SHOT:
        few_shots = sample_few_shot_examples(TRAIN_TSV, FEW_SHOT_ERR_CNT, FEW_SHOT_NOT_CNT, RANDOM_STATE)

    # 5) Batched generation
    gen_texts = []
    preds = []

    # Precompute eos ids list so we also stop at Llama's <|eot_id|>
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            eos_ids.append(eot_id)
    except Exception:
        pass

    total = len(df)
    for start in tqdm(range(0, total, BATCH_SIZE), desc=f"{'Few-shot' if USE_FEW_SHOT else 'Zero-shot'} (Llama 3.1-8B, eager)"):
        end = min(start + BATCH_SIZE, total)
        rows = df.iloc[start:end]

        # Build chat prompts (strings) using official template
        prompts = []
        for _, r in rows.iterrows():
            msgs = build_messages_few_shot(few_shots, r.src, r.mt) if USE_FEW_SHOT else build_messages_zero_shot(r.src, r.mt)
            s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompts.append(s)

        # Fast tokenizer batch path
        enc = tokenizer(
            prompts,
            padding=True,             # left padding already configured
            return_tensors="pt"
        )
        input_ids    = enc["input_ids"].to(model.device)
        attention_ms = enc["attention_mask"].to(model.device)
        Lmax         = input_ids.size(1)   # all prompts padded to this length

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_ms,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode only the newly generated tokens (positions after Lmax)
        for row_idx in range(outputs.size(0)):
            gen_ids = outputs[row_idx, Lmax:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if not text:  # fallback
                text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            gen_texts.append(text)
            lbl = extract_label(text)
            preds.append(1 if lbl == "ERR" else 0)

    # 6) First 10 preview
    print("\nFirst 10 (generated → true / pred):\n")
    for i in range(min(10, len(df))):
        gt = gen_texts[i] if gen_texts[i] else "<EMPTY>"
        true = "ERR" if df.loc[i,'label_id']==1 else "NOT"
        pred = "ERR" if preds[i]==1 else "NOT"
        print(f"#{i+1:2d} Generated: {gt!r}")
        print(f"    True / Pred: {true} / {pred}\n")

    # 7) Metrics
    labels = df["label_id"].tolist()
    mcc    = matthews_corrcoef(labels, preds)
    prf    = precision_recall_fscore_support(labels, preds, labels=[1,0], zero_division=0)
    cm     = confusion_matrix(labels, preds, labels=[1,0])

    print(f"\nMCC   : {mcc:.4f}")
    print(f"F1-ERR: {prf[2][0]:.4f}")
    print(f"F1-NOT: {prf[2][1]:.4f}\n")
    print("Confusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"NOT {cm[1,0]:6d} {cm[1,1]:6d}")

if __name__ == "__main__":
    main()
