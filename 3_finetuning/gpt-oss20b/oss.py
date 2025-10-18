#!/usr/bin/env python3
"""Fine-tune the GPT-OSS 20B model for critical error detection using LoRA."""

# GPT-OSS-20B CED — BF16 LoRA with Unsloth (local-only, render-first dataset)

import os, glob
from typing import List, Dict, Any
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# =================== Config (env overrides optional) ==========================
MODEL_ID   = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")   # repo id OR absolute local path
HF_HOME    = os.environ.get("HF_HOME", "/path/to/local/hf_cache")
DEFAULT_LOCAL = os.path.join(HF_HOME, "models", MODEL_ID.replace("/", "_"))

TRAIN_TSV  = os.environ.get("TRAIN_TSV", "/path/to/datasets/combined_ende_train.tsv")
OUT_DIR    = os.environ.get("OUT_DIR",   "/path/to/outputs/ced_oss20b_lora")

MAX_SEQ_LEN  = int(os.environ.get("MAX_SEQ_LEN", "2048"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACC     = int(os.environ.get("GRAD_ACC", "8"))
LR           = float(os.environ.get("LR", "1.5e-4"))
EPOCHS       = float(os.environ.get("EPOCHS", "2.0"))
LORA_R       = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = 0.0                                   # fast path for Unsloth
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
LOG_STEPS    = int(os.environ.get("LOG_STEPS", "20"))
SAVE_STEPS   = int(os.environ.get("SAVE_STEPS", "1000"))
SEED         = int(os.environ.get("SEED", "42"))
MERGE_FULL_FP16 = os.environ.get("MERGE_FULL_FP16", "0") == "1"

SYSTEM_PROMPT = (
    "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
    "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
    "or only has minor imperfections.\n"
    "Do not add any explanation, punctuation, or additional text."
)

# =================== Data helpers ============================================
def load_tsv_noheader(path: str) -> pd.DataFrame:
    """Load a TSV without headers and normalize to ERR/NOT labels."""
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, na_filter=False)
    n = df.shape[1]
    if n >= 5:
        df = df.iloc[:, :5]; df.columns = ["id","src","mt","raw","label"]
    elif n == 4:
        df.columns = ["src","mt","raw","label"]; df.insert(0,"id",range(len(df)))
    elif n == 3:
        df.columns = ["src","mt","label"]; df.insert(0,"id",range(len(df))); df.insert(3,"raw","")
    else:
        raise ValueError(f"Unexpected TSV columns: {n}")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(["ERR","NOT"])].reset_index(drop=True)
    return df[["src","mt","label"]]

def df_to_messages(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert dataframe rows to chat message dictionaries."""
    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            "messages": [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":f"EN: {r.src.strip()}\nDE: {r.mt.strip()}"},
                {"role":"assistant","content":r.label.strip().upper()},
            ]
        })
    return rows

# =================== Core =====================================================
def _resolve_local_model_path(model_id: str) -> str:
    """Return a usable local snapshot directory for the base model."""
    if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "config.json")):
        return model_id
    candidate = DEFAULT_LOCAL
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
        return candidate
    raise FileNotFoundError(
        f"Could not find a local model snapshot. Tried:\n- {model_id}\n- {candidate}\n"
        "Ensure the snapshot exists locally (for example, under your HF cache directory)."
    )

def _assert_has_weights(local_dir: str):
    """Verify that at least one weight shard is present in ``local_dir``."""
    pats = ["*.safetensors", "*.bin"]
    found = []
    for p in pats:
        found.extend(glob.glob(os.path.join(local_dir, p)))
        found.extend(glob.glob(os.path.join(local_dir, "**", p), recursive=True))
    if not found:
        raise FileNotFoundError(
            f"No model weight files found under {local_dir}. "
            "Expected at least one *.safetensors or *.bin shard."
        )

def main():
    """Prepare data, configure LoRA training, and launch supervised fine-tuning."""
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    model_path = _resolve_local_model_path(MODEL_ID)
    _assert_has_weights(model_path)

    # ---- Load base model/tokenizer (BF16, NO 4-bit) ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,          # absolute local path
        max_seq_length = MAX_SEQ_LEN,
        dtype = torch.bfloat16,
        load_in_4bit = False,             # Plan A
        trust_remote_code = True,
        local_files_only = True,
        attn_implementation = "eager",
        token = None,
    )

    # ---- Build training texts using tokenizer.apply_chat_template ----
    train_df   = load_tsv_noheader(TRAIN_TSV)
    train_rows = df_to_messages(train_df)

    # Render *full* message lists → text (no per-message calls!)
    train_texts = [
        tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        for ex in train_rows
    ]
    train_ds = Dataset.from_dict({"text": train_texts})

    # ---- LoRA (fast path) ----
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,      # 0.0 for Unsloth fast patch
        target_modules = "all-linear",
        bias = "none",
        use_rslora = False,
        loftq_config = None,
    )

    # ---- Trainer ----
    training_args = SFTConfig(
        output_dir = OUT_DIR,
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACC,
        learning_rate = LR,
        warmup_ratio = WARMUP_RATIO,
        weight_decay = WEIGHT_DECAY,
        logging_steps = LOG_STEPS,
        save_steps = SAVE_STEPS,
        bf16 = True,
        lr_scheduler_type = "cosine",
        gradient_checkpointing = True,
        max_seq_length = MAX_SEQ_LEN,
        report_to = "none",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        dataset_text_field = "text",      # <— use pre-rendered text
        args = training_args,
        packing = False,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    if MERGE_FULL_FP16:
        from unsloth import FastLanguageModel as FLM
        merged_dir = os.path.join(OUT_DIR, "merged_fp16")
        os.makedirs(merged_dir, exist_ok=True)
        FLM.merge_and_unload(
            model = model,
            save_dir = merged_dir,
            tokenizer = tokenizer,
            dtype = torch.float16,
        )
        print(f"[OK] Merged full model saved to: {merged_dir}")

if __name__ == "__main__":
    main()
