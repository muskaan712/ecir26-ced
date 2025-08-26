#!/usr/bin/env python3
# few_shot_gpt4o_inference.py

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
    filename="inference_gpt4.1_few_shot.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR         = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV          = os.path.join(DATA_DIR, "ende_majority_dev.tsv")
TRAIN_TSV        = os.path.join(DATA_DIR, "ende_majority_train.tsv")
openai.api_key   = "sk-proj-eOGkRdhQf1kLt0eOhMAaT3BlbkFJ3cKHnSymxwTdakeiMwze"  # ensure this is set
MODEL            = "gpt-4.1"
# Only need minimal completion tokens for label
MAX_TOKENS       = 3
TEMPERATURE      = 0.0
# Cost estimation parameters (USD per 1K tokens)
COST_PER_1K      = 0.002
# Few-shot example counts to address class imbalance (5 ERR, 3 NOT)
FEW_SHOT_ERR_CNT = 5
FEW_SHOT_NOT_CNT = 3
# ────────────────────────────────────────────────────────────────────────────────

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


def select_few_shot_examples():
    """Sample examples oversampling ERR to address class imbalance (5 ERR + 3 NOT)."""
    train_df = load_data(TRAIN_TSV)
    # Oversample ERR minority class
    err_examples = train_df[train_df.label == "ERR"].sample(FEW_SHOT_ERR_CNT, random_state=42)
    not_examples = train_df[train_df.label == "NOT"].sample(FEW_SHOT_NOT_CNT, random_state=42)
    examples = []
    # ERR examples
    for _, row in err_examples.iterrows():
        examples.append({
            "src": row.src.strip(),
            "mt": row.mt.strip(),
            "label": "ERR"
        })
    # NOT examples
    for _, row in not_examples.iterrows():
        examples.append({
            "src": row.src.strip(),
            "mt": row.mt.strip(),
            "label": "NOT"
        })
    logging.info(f"Loaded {len(err_examples)} ERR and {len(not_examples)} NOT few-shot examples.")
    return examples


def build_messages(src: str, mt: str, examples):
    """Construct chat messages for classification tasks."""
    system_prompt = (
        "You are a precise translation evaluator.\n"
        "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: 'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate or only has minor imperfections.\n"
        "Do not add any explanation, punctuation, or additional text."
    )
    messages = [{"role": "system", "content": system_prompt}]
    # Add few-shot examples in order: all ERR then NOT
    for ex in examples:
        messages.append({"role": "user", "content": f"EN: {ex['src']}\nDE: {ex['mt']}"})
        messages.append({"role": "assistant", "content": ex['label']})
    # Add the query at the end
    messages.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}"})
    return messages


def main():
    """Run inference over the dataset and report metrics."""
    # Load dev set and few-shot examples
    df = load_data(DEV_TSV)
    examples = select_few_shot_examples()

    gen_labels = []
    preds = []
    total_tokens = 0
    start_time = time.time()

    # Inference loop
    for i, row in tqdm(df.iterrows(), total=len(df), desc="GPT-4o few-shot inference"):
        messages = build_messages(row.src, row.mt, examples)
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=1
        )
        label = resp.choices[0].message.content.strip().upper()
        gen_labels.append(label)
        usage = resp.usage
        total_tokens += usage.prompt_tokens + usage.completion_tokens
        pred = 1 if label == 'ERR' else 0
        preds.append(pred)
        logging.info(
            f"Row {i}: generated={label}, prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}"
        )

    # Log total time and cost
    end_time = time.time()
    logging.info(f"Total inference time: {end_time - start_time:.2f}s")
    logging.info(f"Total tokens used: {total_tokens}")
    cost = (total_tokens / 1000) * COST_PER_1K
    logging.info(f"Estimated cost: ${cost:.4f}")

    # Preview first 10 outputs
    print("\nFirst 10 results (Generated | True | Pred):\n")
    for i in range(10):
        gen = gen_labels[i] or "<EMPTY>"
        true = 'ERR' if df.loc[i, 'label_id'] == 1 else 'NOT'
        pred = 'ERR' if preds[i] == 1 else 'NOT'
        print(f"#{i+1:2d} {gen!r} | {true} | {pred}")

    # Compute and print metrics
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
