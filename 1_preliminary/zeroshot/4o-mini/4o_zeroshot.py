#!/usr/bin/env python3
# zero_shot_gpt4o_inference.py

import os
import pandas as pd
from tqdm import tqdm
import time
import logging
import openai
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix
)

# Set up logging
logging.basicConfig(
    filename="inference_gpt4o_zero_shot.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = os.path.join(DATA_DIR, "ende_majority_dev.tsv")
# Ensure your OpenAI API key is set in the environment

openai.api_key = os.getenv("OAPI")
MODEL          = "gpt-4o-mini"
# Only need minimal completion tokens because output is one token
MAX_TOKENS     = 3
TEMPERATURE    = 0.0
# Cost estimation parameters (in USD per 1K tokens)
COST_PER_1K    = 0.002
# ────────────────────────────────────────────────────────────────────────────────

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


def build_messages(src: str, mt: str):
    # Optimized system prompt for GPT-4o classification
    system_prompt = (
        "You are a precise translation evaluator.\n"
        "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: 'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate or only has minor imperfections.\n"
        "Do not add any explanation, punctuation, or additional text."
    )
    user_prompt = f"EN: {src.strip()}\nDE: {mt.strip()}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def main():
    df = load_data(DEV_TSV)
    gen_labels = []
    preds = []
    total_tokens = 0
    start_time = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="GPT-4o zero-shot inference"):
        messages = build_messages(row.src, row.mt)
        # Call without stop to ensure label is returned
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=1
        )
        text = resp.choices[0].message.content.strip().upper()
        gen_labels.append(text)
        usage = resp.usage
        total_tokens += usage.prompt_tokens + usage.completion_tokens
        pred = 1 if text == 'ERR' else 0
        preds.append(pred)
        logging.info(
            f"Row {i}: generated={text}, prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}"
        )

    end_time = time.time()
    logging.info(f"Total inference time: {end_time - start_time:.2f}s")
    logging.info(f"Total tokens used: {total_tokens}")
    cost = (total_tokens / 1000) * COST_PER_1K
    logging.info(f"Estimated cost: ${cost:.4f}")

    # Preview first 10 outputs
    print("\nFirst 10 results:\n")
    for i in range(10):
        gen = gen_labels[i] or "<EMPTY>"
        true = 'ERR' if df.loc[i, 'label_id'] == 1 else 'NOT'
        pred = 'ERR' if preds[i] == 1 else 'NOT'
        print(f"#{i+1:2d} Generated: {gen!r} | True: {true} | Pred: {pred}")

    # Compute metrics
    labels = df['label_id'].tolist()
    mcc = matthews_corrcoef(labels, preds)
    prf = precision_recall_fscore_support(labels, preds, labels=[1,0], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[1,0])

    print(f"\nMCC   : {mcc:.4f}")
    print(f"F1-ERR: {prf[2][0]:.4f}")
    print(f"F1-NOT: {prf[2][1]:.4f}\n")
    print("Confusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"NOT {cm[1,0]:6d} {cm[1,1]:6d}")

if __name__ == "__main__":
    main()
