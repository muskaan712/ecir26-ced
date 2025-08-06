#!/usr/bin/env python3
# zero_shot_llama31_8b_improved_v3.py

import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline
)
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix
)

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = os.path.join(DATA_DIR, "ende_majority_dev.tsv")

HF_TOKEN       = "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"
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.3"  # Swapped to Mistral 7B-Instruct
CACHE_ROOT     = "modelcache"
CACHE_DIR      = os.path.join(CACHE_ROOT, MODEL_ID.replace("/", "_"))

BATCH_SIZE     = 8
MAX_NEW_TOKENS = 30    # Increased for more reasoning space before final label
TEMPERATURE    = 0.0   # Deterministic to reduce variability
# ────────────────────────────────────────────────────────────────────────────────

def download_and_cache_model():
    """Download MODEL_ID into CACHE_DIR once, then reuse."""
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=CACHE_DIR,
            token=HF_TOKEN
        )

def load_data(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["id","src","mt","toklabels","label"],
        dtype={"id": str}
    )
    df["label_id"] = df["label"].map({"ERR": 1, "NOT": 0})
    return df

def make_prompt(src: str, mt: str) -> str:
    # Stricter criteria for ERR, expanded step-by-step for balance
    system_prompt = (
        "You are a precise translation evaluator. For the English source (EN) and German translation (DE), "
        "check if DE accurately captures EN's meaning. Label 'ERR' ONLY for clear errors like major meaning shifts, "
        "factual inaccuracies, or omissions. Label 'NOT' for accurate or minorly imperfect translations. "
        "Think step-by-step: 1) Summarize EN meaning. 2) Summarize DE meaning. 3) Compare for alignment. "
        "4) Decide ERR only if mismatched, else NOT. Reply ONLY with 'ERR' or 'NOT' at the end."
    )
    
    user_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"EN: {src.strip()}\n"
        f"DE: {mt.strip()}\n"
        f"Follow the steps and reply ONLY with ERR or NOT.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # Combine: BOS token + system + user
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>\n\n"
        f"{user_prompt}"
    )

def extract_text(out):
    if isinstance(out, dict) and "generated_text" in out:
        return out["generated_text"]
    if isinstance(out, str):
        return out
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            return first.get("generated_text", "")
        if isinstance(first, str):
            return first
    return ""

def main():
    # 1) Cache model locally
    download_and_cache_model()

    # 2) Load data & prompts
    df = load_data(DEV_TSV)
    prompts = [make_prompt(r.src, r.mt) for _, r in df.iterrows()]

    # 3) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        CACHE_DIR, use_fast=True, local_files_only=True
    )
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    model = AutoModelForCausalLM.from_pretrained(
        CACHE_DIR,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 4) Create pipeline
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,  # Deterministic
        return_full_text=False
    )

    # 5) Inference with guard against empty gen
    gen_texts = []
    preds = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Improved zero-shot (Llama 3.1-8B)"):
        batch = prompts[i : i + BATCH_SIZE]
        outputs = pipe(batch)
        for out in outputs:
            gen = extract_text(out).strip().upper()
            gen_texts.append(gen)
            # Take the last occurrence of ERR/NOT to focus on final decision
            words = gen.split()
            last_label = [w for w in reversed(words) if w in ('ERR', 'NOT')]
            if last_label:
                pred = 1 if last_label[0] == 'ERR' else 0
            else:
                pred = 0  # Default to NOT for ambiguity/bias correction
            preds.append(pred)

    # 6) Preview first 10 examples
    print("\nFirst 10 (generated text → true / pred):\n")
    for i in range(10):
        gt = gen_texts[i] or "<EMPTY>"
        true = "ERR" if df.loc[i,'label_id']==1 else "NOT"
        pred = "ERR" if preds[i]==1 else "NOT"
        print(f"#{i+1:2d} Generated: {gt!r}")
        print(f"    True / Pred: {true} / {pred}\n")

    # 7) Compute metrics
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
