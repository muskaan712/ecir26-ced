#!/usr/bin/env python3
# zero_shot_llama31_8b_batch_generate.py
#
# Zero-shot CED (EN→DE) with Meta-Llama-3.1-8B-Instruct
# - Caches model to ./modelcache/meta-llama_Meta-Llama-3.1-8B-Instruct
# - Uses Llama 3.1 chat template correctly (tokenized)
# - Batched manual generate (no TextGenerationPipeline)
# - Forces eager attention only (no FlashAttention, no SDPA)
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

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = "/home/s13mchop/LLMs/data/wmt21/ende_majority_dev.tsv"

HF_TOKEN       = os.getenv("HF_TOKEN")  # set and accept license on HF
MODEL_ID       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_ROOT     = "modelcache"
CACHE_DIR      = os.path.join(CACHE_ROOT, MODEL_ID.replace("/", "_"))

BATCH_SIZE     = 8
MAX_NEW_TOKENS = 3      # only need 'ERR' or 'NOT'
DO_SAMPLE      = False  # deterministic

# Force eager kernels only (no FA2, no SDPA)
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
os.environ.pop("USE_FLASH_ATTENTION_2", None)
# ──────────────────────────────────────────────────────────────────────────────

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

def load_data(path):
    """Load TSV data into a DataFrame and map labels to numeric IDs."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["id","src","mt","toklabels","label"],
        dtype={"id": str}
    )
    df["label_id"] = df["label"].map({"ERR": 1, "NOT": 0})
    return df

def build_messages(src: str, mt: str):
    """Llama 3.1 chat messages: system + user."""
    system_prompt = (
        "You are a precise translation evaluator.\n"
        "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
        "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
        "or only has minor imperfections.\n"
        "Do not add any explanation, punctuation, or additional text."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}\nAnswer with ONLY 'ERR' or 'NOT'."}
    ]

def extract_label(text: str) -> str:
    """Extract final 'ERR' or 'NOT'. Default to NOT if ambiguous/empty."""
    t = (text or "").upper()
    matches = re.findall(r"\b(ERR|NOT)\b", t)
    return matches[-1] if matches else "NOT"

def main():
    """Run inference over the dataset and report metrics."""
    # 1) Cache locally
    download_and_cache_model()

    # 2) Load tokenizer/model (eager only)
    tokenizer = AutoTokenizer.from_pretrained(CACHE_DIR, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only batching best practice

    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        CACHE_DIR,
        local_files_only=True,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="eager"  # explicit; no SDPA/FA2
    )
    model.eval()

    # 3) Load data
    df = load_data(DEV_TSV)

    # 4) Build tokenized prompts with the official chat template
    #    We collect prompt tensors and their original lengths for slicing new tokens later.
    prompts_input_ids = []
    prompts_attn_mask = []
    prompt_lens = []

    for _, r in df.iterrows():
        msgs = build_messages(r.src, r.mt)
        encoded = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt"   # returns tokenized tensors directly
        )
        # encoded is a 1 x L tensor (input_ids)
        input_ids = encoded
        attn_mask = torch.ones_like(input_ids)

        prompts_input_ids.append(input_ids)
        prompts_attn_mask.append(attn_mask)
        prompt_lens.append(input_ids.shape[1])

    # 5) Batched generate
    gen_texts = []
    preds = []

    device = model.device
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Zero-shot (Llama 3.1-8B, eager)"):
        batch_ids   = prompts_input_ids[i:i+BATCH_SIZE]
        batch_masks = prompts_attn_mask[i:i+BATCH_SIZE]
        batch_lens  = prompt_lens[i:i+BATCH_SIZE]

        # pad to same length (left padding set above)
        padded = tokenizer.pad(
            {"input_ids": [x[0] for x in batch_ids], "attention_mask": [m[0] for m in batch_masks]},
            padding=True,
            return_tensors="pt"
        )
        input_ids    = padded["input_ids"].to(device)
        attention_ms = padded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_ms,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                min_new_tokens=1,                         # <- ensure at least one token
                eos_token_id=tokenizer.eos_token_id,      # stop at chat EOT
                pad_token_id=tokenizer.pad_token_id
            )

        # Slice per-row new tokens based on each row's prompt length
        for row_idx in range(outputs.size(0)):
            plen  = batch_lens[row_idx]
            # Because we padded, individual prompts may start later; compute the true start:
            # Find the index of the first non-pad token in the row
            row_input = input_ids[row_idx]
            non_pad_start = (row_input != tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
            # Effective prompt length is row length from non_pad_start to end
            eff_prompt_len = row_input.size(0) - non_pad_start

            # The generated output row:
            out_ids = outputs[row_idx]

            # Slice only the new tokens AFTER the full prompt length (eff_prompt_len)
            gen_ids = out_ids[eff_prompt_len:]

            # Decode; try skipping specials first, then fallback without skipping
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if not decoded:
                decoded = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

            gen_texts.append(decoded)
            label = extract_label(decoded)
            preds.append(1 if label == "ERR" else 0)

    # 6) Preview first 10
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
