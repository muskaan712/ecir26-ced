#!/usr/bin/env python3
# ft_llama31_8b_nodev.py — Llama 3.1 8B CED SFT (local-only, no env vars, no downloads)

import os, math, random
import pandas as pd
import torch

# Unsloth first
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# ── Fixed paths ───────────────────────────────────────────────────────────────
TRAIN_TSV  = "/home/ni124545/llm/data/combined_ende_train.tsv"
CACHE_DIR  = "/hpcwork/ni124545/hf_cache/models/meta-llama_Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "/hpcwork/ni124545/ced_runs/llama31_8b/latest"

# ── Training knobs ────────────────────────────────────────────────────────────
EPOCHS         = 2
BATCH_SIZE     = 2          # per GPU
GRAD_ACCUM     = 8
LR             = 2e-4
MAX_SEQ_LEN    = 2048
WARMUP_RATIO   = 0.03
WEIGHT_DECAY   = 0.0
LOG_STEPS      = 25
SAVE_STEPS     = 1000
PACKING        = True
SEED           = 42
MERGE_WEIGHTS  = True
MAX_TRAIN_ROWS = ""         # "" = all
USE_4BIT       = True       # auto-fallback to BF16 if bnb not available

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a STRICT binary classifier for WMT’21 Task 3 (Critical Error Detection, EN→DE).\n"
    "Output EXACTLY one token: ERR or NOT (UPPERCASE, no punctuation, no explanation).\n\n"
    "ERR if ANY critical meaning deviation exists:\n"
    "• TOX • SAF • NAM • SEN • NUM\n\n"
    "NOT for non-critical issues. If uncertain, choose NOT.\n"
    "Return ONLY: ERR or NOT."
)
USER_TEMPLATE = "EN: {src}\nDE: {mt}\nAnswer with ONLY 'ERR' or 'NOT'."

# ── TSV utils ─────────────────────────────────────────────────────────────────
def _map_label_to_err_not(x: str) -> str:
    s = str(x or "").strip().upper()
    if s == "BAD": return "ERR"
    if s == "OK":  return "NOT"
    if s in ("ERR","NOT"): return s
    if "ERR" in s: return "ERR"
    if "NOT" in s: return "NOT"
    return "ERR"

def _split_tsv_flex(path: str):
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, na_filter=False)
    n = df.shape[1]
    if n < 3: raise ValueError(f"Expected ≥3 columns (src, mt, label). Got {n} in {path}")
    if n == 3:
        src, mt, label = df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]
    elif n == 4:
        src, mt, label = df.iloc[:,0], df.iloc[:,1], df.iloc[:,3]
    else:
        src, mt, label = df.iloc[:,1], df.iloc[:,2], df.iloc[:,4]
    return src.astype(str), mt.astype(str), label.astype(str).map(_map_label_to_err_not)

def load_train_df(path: str) -> pd.DataFrame:
    src, mt, label = _split_tsv_flex(path)
    df = pd.DataFrame({"src": src, "mt": mt, "label": label})
    df = df[df["label"].isin(["ERR","NOT"])].reset_index(drop=True)
    if MAX_TRAIN_ROWS.isdigit():
        df = df.sample(int(MAX_TRAIN_ROWS), random_state=SEED).reset_index(drop=True)
    return df

def df_to_text_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    def row_to_chat(r):
        return [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":USER_TEMPLATE.format(src=r["src"].strip(), mt=r["mt"].strip())},
            {"role":"assistant","content":r["label"].strip().upper()},
        ]
    chats = [{"messages": row_to_chat(r)} for _, r in df.iterrows()]
    ds = Dataset.from_list(chats)
    return ds.map(lambda ex: {
        "text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
    })

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # No HF_* envs at all
    for k in ("HF_HOME","TRANSFORMERS_CACHE","HF_TOKEN","HF_HUB_ENABLE_HF_TRANSFER"):
        os.environ.pop(k, None)

    random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tokenizer & model strictly from local files
    tokenizer = AutoTokenizer.from_pretrained(CACHE_DIR, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.use_default_system_prompt = False

    load_in_4bit = False
    if USE_4BIT:
        try:
            import bitsandbytes as _  # noqa
            load_in_4bit = True
            print("[quant] Using 4-bit QLoRA.")
        except Exception as e:
            print(f"[quant] bitsandbytes unavailable ({e}) → BF16 fallback.")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=CACHE_DIR,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
        local_files_only=True,
        token=None,
    )

    # Use explicit Llama 3.x linear modules + dropout=0.0 for Unsloth fast path
    LLAMA_LORA_TARGETS = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,          # enables Unsloth fast patch
        target_modules=LLAMA_LORA_TARGETS,
        use_rslora=False,
        loftq_config=None,
    )

    print("[data] TRAIN:", TRAIN_TSV)
    train_df   = load_train_df(TRAIN_TSV)
    train_text = df_to_text_dataset(train_df, tokenizer)

    world = max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(train_text) / (BATCH_SIZE * world * GRAD_ACCUM))
    print(f"[train] steps/epoch≈{steps_per_epoch} | examples={len(train_text)} | world={world}")

    # ⚠️ No evaluation args — your version of Transformers rejects evaluation_strategy
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        bf16=(not load_in_4bit),
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit" if load_in_4bit else "adamw_torch",
        dataloader_pin_memory=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=PACKING,
        args=args,
    )

    print("[train] SFT starting …")
    trainer.train()

    print("[save] Saving LoRA + tokenizer …")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
    tokenizer.save_pretrained(OUTPUT_DIR)

    if MERGE_WEIGHTS:
        try:
            print("[merge] Merging LoRA → base …")
            merged = FastLanguageModel.merge_and_unload(model, tokenizer)
            merged.save_pretrained(os.path.join(OUTPUT_DIR, "merged"), safe_serialization=True)
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "merged"))
            print("[merge] Full merged model at:", os.path.join(OUTPUT_DIR, "merged"))
        except Exception as e:
            print(f"[merge] Merge skipped: {e}")

    print("[done] Training complete.")

if __name__ == "__main__":
    main()
