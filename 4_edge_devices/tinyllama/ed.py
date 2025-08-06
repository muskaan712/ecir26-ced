#!/usr/bin/env python3
# few_shot_tinyllama_optimized.py

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
import time
import logging

# Set up logging
logging.basicConfig(filename="inference.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = os.path.join(DATA_DIR, "ende_majority_dev.tsv")
TRAIN_TSV      = os.path.join(DATA_DIR, "ende_majority_train.tsv")

HF_TOKEN       = os.getenv("HF_TOKEN")  # ensure this is set
MODEL_ID       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CACHE_ROOT     = "modelcache"
CACHE_DIR      = os.path.join(CACHE_ROOT, MODEL_ID.replace("/", "_"))

BATCH_SIZE     = 8
MAX_NEW_TOKENS = 10    # Reduced since we only need "ERR" or "NOT"
TEMPERATURE    = 0.0   # Deterministic

# Cost estimation parameters
COST_PER_1K_TOKENS = 0.002
# ────────────────────────────────────────────────────────────────────────────────

def download_and_cache_model():
    """Download MODEL_ID into CACHE_DIR once, then reuse."""
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=CACHE_DIR,
            token=HF_TOKEN,
            max_workers=12
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

def select_examples_from_training():
    """Select oversampled examples from training data to address class imbalance (5 ERR + 3 NOT)."""
    try:
        train_df = pd.read_csv(
            TRAIN_TSV,
            sep="\t",
            header=None,
            names=["id","src","mt","toklabels","label"],
            dtype={"id": str}
        )
        err_examples = train_df[train_df['label'] == 'ERR'].sample(5, random_state=42)
        not_examples = train_df[train_df['label'] == 'NOT'].sample(3, random_state=42)
        examples = []
        
        # Add ERR examples
        for _, row in err_examples.iterrows():
            examples.append(
                f"[INST] EN: {row['src']}\nDE: {row['mt']}\nReply ONLY with ERR or NOT. [/INST] ERR </s>"
            )
        
        # Add NOT examples
        for _, row in not_examples.iterrows():
            examples.append(
                f"[INST] EN: {row['src']}\nDE: {row['mt']}\nReply ONLY with ERR or NOT. [/INST] NOT </s>"
            )
        
        print(f"✓ Loaded {len(err_examples)} ERR and {len(not_examples)} NOT examples from training data")
        return " ".join(examples)  # Concatenate as a chain for few-shot
        
    except Exception as e:
        print(f"⚠ Warning: Could not load training data: {e}")
        print("⚠ Falling back to hardcoded examples with ERR oversampling")
        return (
            "[INST] EN: This is completely wrong information.\nDE: Das ist völlig falsche Information.\nReply ONLY with ERR or NOT. [/INST] ERR </s> "
            "[INST] EN: The translation has major errors.\nDE: Die Übersetzung hat große Fehler.\nReply ONLY with ERR or NOT. [/INST] ERR </s> "
            "[INST] EN: Significant meaning distortion here.\nDE: Bedeutende Bedeutungsverzerrung hier.\nReply ONLY with ERR or NOT. [/INST] ERR </s> "
            "[INST] EN: Critical information is missing.\nDE: Kritische Informationen fehlen.\nReply ONLY with ERR or NOT. [/INST] ERR </s> "
            "[INST] EN: Major factual mistake present.\nDE: Großer Sachfehler vorhanden.\nReply ONLY with ERR or NOT. [/INST] ERR </s> "
            "[INST] EN: This is a good translation.\nDE: Das ist eine gute Übersetzung.\nReply ONLY with ERR or NOT. [/INST] NOT </s> "
            "[INST] EN: The meaning is preserved well.\nDE: Die Bedeutung ist gut erhalten.\nReply ONLY with ERR or NOT. [/INST] NOT </s> "
            "[INST] EN: Only minor issues here.\nDE: Nur kleinere Probleme hier.\nReply ONLY with ERR or NOT. [/INST] NOT </s>"
        )

def make_prompt(src: str, mt: str) -> str:
    system_prompt = (
        "You are a precise translation evaluator. Given the English source (EN) and German translation (DE), "
        "determine if DE accurately reflects EN's meaning. Output 'ERR' ONLY for significant errors such as "
        "meaning distortions, factual mistakes, or key omissions. Output 'NOT' for translations that are accurate "
        "or have only minor flaws. Your response must be exactly 'ERR' or 'NOT'—nothing else, no explanations."
    )
    few_shot = select_examples_from_training()
    user_prompt = f"[INST] EN: {src.strip()}\nDE: {mt.strip()}\nReply ONLY with ERR or NOT. [/INST]"
    
    return (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{few_shot} {user_prompt}"
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
    download_and_cache_model()

    # Enable TinyLlama fused kernels
    os.environ.setdefault("TINY_FUSED_RMSNORM", "1")
    os.environ.setdefault("TINY_FUSED_CROSSENTROPY", "1")
    os.environ.setdefault("TINY_FUSED_ROTARY", "1")
    os.environ.setdefault("TINY_FUSED_SWIGLU", "1")

    df = load_data(DEV_TSV)
    prompts = [make_prompt(r.src, r.mt) for _, r in df.iterrows()]

    tokenizer = AutoTokenizer.from_pretrained(
        CACHE_DIR, use_fast=True, local_files_only=True
    )
    # Remove the pad_token addition as Llama-2 tokenizers handle EOS fine
    # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Attempt FlashAttention2, else fallback
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CACHE_DIR,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        print("✓ Using FlashAttention2")
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            CACHE_DIR,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("✓ Using standard attention")

    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,
        return_full_text=False
    )

    gen_texts, preds = [], []
    total_tokens, start_time = 0, time.time()

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="TinyLlama few-shot optimized"):
        batch = prompts[i: i + BATCH_SIZE]
        batch_start = time.time()
        outputs = pipe(batch)
        logging.info(f"Batch {i//BATCH_SIZE + 1}: size={len(batch)}, latency={time.time() - batch_start:.3f}s")
        
        for out, prompt in zip(outputs, batch):
            gen = extract_text(out).strip().upper()
            gen_texts.append(gen)
            
            # Token counting for cost estimation
            in_tokens = len(tokenizer.encode(prompt))
            out_tokens = len(tokenizer.encode(gen))
            total_tokens += in_tokens + out_tokens
            
            # Improved prediction logic - look for ERR or NOT in the generated text
            if 'ERR' in gen and 'NOT' not in gen:
                preds.append(1)  # ERR
            elif 'NOT' in gen and 'ERR' not in gen:
                preds.append(0)  # NOT
            elif 'ERR' in gen and 'NOT' in gen:
                # If both are present, take the first one that appears
                err_pos = gen.find('ERR')
                not_pos = gen.find('NOT')
                preds.append(1 if err_pos < not_pos else 0)
            else:
                # Default to NOT if neither is clearly present
                preds.append(0)

    logging.info(f"Total tokens: {total_tokens}, total time: {time.time() - start_time:.3f}s")
    cost_estimate = (total_tokens / 1000) * COST_PER_1K_TOKENS
    logging.info(f"Estimated cost: ${cost_estimate:.4f}")

    # Metrics & display
    print("\nFirst 10 results:")
    for i in range(10):
        gt = gen_texts[i] or "<EMPTY>"
        true = "ERR" if df.loc[i,'label_id']==1 else "NOT"
        pred = "ERR" if preds[i]==1 else "NOT"
        print(f"#{i+1:2d} Generated: {gt!r}    True/Pred: {true}/{pred}")

    labels = df["label_id"].tolist()
    mcc = matthews_corrcoef(labels, preds)
    prf = precision_recall_fscore_support(labels, preds, labels=[1,0], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[1,0])

    print(f"\nMCC   : {mcc:.4f}")
    print(f"F1-ERR: {prf[2][0]:.4f}, F1-NOT: {prf[2][1]:.4f}")
    print("Confusion Matrix (true rows x pred cols):")
    print("      ERR   NOT")
    print(f"ERR {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"NOT {cm[1,0]:6d} {cm[1,1]:6d}")

    # Additional metrics
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Total examples: {len(labels)}")
    print(f"ERR examples: {sum(labels)}")
    print(f"NOT examples: {len(labels) - sum(labels)}")

if __name__ == "__main__":
    main()
