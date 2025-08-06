#!/usr/bin/env python3
# few_shot_llama3_8b_best_baseline.py


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
MODEL_ID       = "meta-llama/Meta-Llama-3-8B-Instruct"  # Llama 3 8B Instruct
CACHE_ROOT     = "modelcache"
CACHE_DIR      = os.path.join(CACHE_ROOT, MODEL_ID.replace("/", "_"))


BATCH_SIZE     = 8
MAX_NEW_TOKENS = 50    # For reasoning space before final label
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


def select_examples_from_training():
    """Select oversampled examples from training data to address class imbalance (5 ERR + 3 NOT)."""
    try:
        # Load training data using same format as dev data
        train_df = pd.read_csv(
            TRAIN_TSV,
            sep="\t",
            header=None,
            names=["id","src","mt","toklabels","label"],
            dtype={"id": str}
        )
        
        # Oversample ERR (minority class) to balance few-shot examples
        err_examples = train_df[train_df['label'] == 'ERR'].sample(5, random_state=42)
        not_examples = train_df[train_df['label'] == 'NOT'].sample(3, random_state=42)
        
        examples = []
        
        # Add ERR examples
        for _, row in err_examples.iterrows():
            examples.append(
                f"<|start_header_id|>user<|end_header_id|>\nEN: {row['src']}\nDE: {row['mt']}\nReply ONLY with ERR or NOT.<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\nERR<|eot_id|>\n"
            )
        
        # Add NOT examples
        for _, row in not_examples.iterrows():
            examples.append(
                f"<|start_header_id|>user<|end_header_id|>\nEN: {row['src']}\nDE: {row['mt']}\nReply ONLY with ERR or NOT.<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\nNOT<|eot_id|>\n"
            )
        
        print(f"✓ Loaded {len(err_examples)} ERR (oversampled) and {len(not_examples)} NOT examples from training data")
        return "".join(examples)
    
    except Exception as e:
        print(f"⚠ Warning: Could not load training data: {e}")
        print("⚠ Falling back to hardcoded examples with ERR oversampling")
        return (
            # Oversampled ERR examples (5 ERR + 3 NOT hardcoded)
            "<|start_header_id|>user<|end_header_id|>\nEN: LOL yeah good one mate - have the rest of the day off champ\nDE: LOL ja gut ein Freund - haben Sie den Rest des Tages freien Champion\nReply ONLY with ERR or NOT.<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\nERR<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\nEN: She was also being very very mean at the second that I made it\nDE: Sie war auch sehr sehr böse in der Sekunde , dass ich es\nReply ONLY with ERR or NOT.<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\nERR<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\nEN: STOP SENDING ME STUPID MESSAGES. I AM FIXING A MISTAKE IN THE ARTICLE.\nDE: STOP ME STUPID MESSAGES SSEN . Ich FIXE einen Fehler im Artikel .\nReply ONLY with ERR or NOT.<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\nERR<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\nEN: This is a test error sentence.\nDE: Das ist ein Testfehlersatz.\nReply ONLY with ERR or NOT.<|eot_id|>\n"  # Placeholder additional ERR
            "<|start_header_id|>assistant<|end_header_id|>\nERR<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\nEN: Another error example.\nDE: Ein weiteres Fehlerbeispiel.\nReply ONLY with ERR or NOT.<|eot_id|>\n"  # Placeholder additional ERR
            "<|start_header_id|>assistant<|end_header_id|>\nERR<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\nEN: especially since their actions were often on behalf of the establishment they worked for.\nDE: vor allem , weil ihre Handlungen oft im Namen der Einrichtung waren , für die sie arbeiteten .\nReply ONLY with ERR or NOT.<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\nNOT<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\nEN: I totally agree. Their website list Italian ice as their product. [URL]\nDE: Ich stimme dem vollkommen zu . Ihre Website listet italienisches Eis als ihr Produkt . [ URL ]\nReply ONLY with ERR or NOT.<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\nNOT<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\nEN: Phalin maybe worthy of a section depending on the Thai impact while the IDL one should be relegated as it were.\nDE: Phalin kann vielleicht einen Abschnitt würdig sein , abhängig von der thailändischen Wirkung , während die IDL sollte wie es sei vernachlässigt werden .\nReply ONLY with ERR or NOT.<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\nNOT<|eot_id|>\n"
        )


def make_prompt(src: str, mt: str) -> str:
    # System prompt from best performing version
    system_prompt = (
        "You are a precise translation evaluator. For the English source (EN) and German translation (DE), "
        "check if DE accurately captures EN's meaning. Label 'ERR' only for clear errors like major meaning shifts, "
        "factual inaccuracies, or omissions. Label 'NOT' for accurate or minorly imperfect translations. "
        "Reply ONLY with 'ERR' or 'NOT'."
    )
    
    # Use training data examples
    few_shot = select_examples_from_training()
    
    user_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\nEN: {src.strip()}\nDE: {mt.strip()}\nReply ONLY with ERR or NOT.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    # Combine for Llama 3 format
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>\n"
        f"{few_shot}"
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
        do_sample=False,
        return_full_text=False
    )


    # 5) Inference with logging
    gen_texts = []
    preds = []
    total_tokens = 0
    start_time = time.time()


    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Baseline 3+3 few-shot (Llama 3 8B)"):
        batch = prompts[i : i + BATCH_SIZE]
        batch_start = time.time()
        outputs = pipe(batch)
        batch_end = time.time()
        # Log batch latency
        logging.info(f"Batch {i//BATCH_SIZE + 1}: size={len(batch)}, latency={batch_end - batch_start:.3f}s")


        for out, prompt in zip(outputs, batch):
            gen = extract_text(out).strip().upper()
            gen_texts.append(gen)
            # Token counting
            in_tokens = len(tokenizer.encode(prompt))
            out_tokens = len(tokenizer.encode(gen))
            total_tokens += in_tokens + out_tokens
            # Basic parsing (to be enhanced)
            if 'ERR' in gen:
                pred = 1
            elif 'NOT' in gen:
                pred = 0
            else:
                pred = 0  # Default to NOT
            preds.append(pred)


    end_time = time.time()
    total_latency = end_time - start_time
    logging.info(f"Total latency: {total_latency:.3f}s")
    logging.info(f"Total tokens processed: {total_tokens}")


    # Estimated cost
    cost_estimate = (total_tokens / 1000) * COST_PER_1K_TOKENS
    logging.info(f"Estimated cost (based on ${COST_PER_1K_TOKENS}/1K tokens): ${cost_estimate:.4f}")


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
