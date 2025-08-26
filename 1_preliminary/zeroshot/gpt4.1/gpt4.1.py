#!/usr/bin/env python3
"""
zero_shot_gpt4_1_full.py
Evaluates *all* rows of ende_majority_dev.tsv with the gpt-4.1 chat model.
"""

import os, openai, pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    matthews_corrcoef, precision_recall_fscore_support, confusion_matrix,
)

# ─── Config ──────────────────────────────────────────────────────────────
openai.api_key = "sk-proj-eOGkRdhQf1kLt0eOhMAaT3BlbkFJ3cKHnSymxwTdakeiMwze"  # export before running
MODEL_ID       = "gpt-4.1"                            # chat-capable model
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = os.path.join(DATA_DIR, "ende_majority_dev.tsv")
MAX_COMP_TOK   = 8                                    # small safety margin
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: 'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate or only has minor imperfections.\n"
    "Do not add any explanation, punctuation, or additional text."
)

def first_token(text: str) -> str:
    tok = (text or "").strip().split()[:1]
    return tok[0].upper() if tok and tok[0].upper() in {"ERR", "NOT"} else "UNKNOWN"

def load_all(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["id","src","mt","toklabels","label"],
        dtype={"id": str}
    )
    df["label_id"] = df["label"].map({"ERR":1, "NOT":0})
    return df

def main() -> None:
    """Run inference over the dataset and report metrics."""
    df = load_all(DEV_TSV)
    preds, raw_out = [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{MODEL_ID} full-set"):
        resp = openai.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"EN: {row.src.strip()}\nDE: {row.mt.strip()}"},
            ],
            max_completion_tokens=MAX_COMP_TOK,
            n=1,
        )
        raw  = resp.choices[0].message.content
        label = first_token(raw)
        raw_out.append(raw)
        preds.append(1 if label == "ERR" else 0)

    # overall metrics
    y_true = df["label_id"].tolist()
    mcc    = matthews_corrcoef(y_true, preds)
    prf    = precision_recall_fscore_support(y_true, preds, labels=[1,0], zero_division=0)
    cm     = confusion_matrix(y_true, preds, labels=[1,0])

    print("\n=== Metrics on full dev set ===")
    print("MCC    :", mcc)
    print("F1-ERR :", prf[2][0], "F1-NOT :", prf[2][1])
    print("Confusion (rows=true, cols=pred):")
    print("      ERR     NOT")
    print(f"ERR  {cm[0,0]:7d} {cm[0,1]:7d}")
    print(f"NOT  {cm[1,0]:7d} {cm[1,1]:7d}")

if __name__ == "__main__":
    if not openai.api_key:
        raise SystemExit("❌  OPENAI_API_KEY not set in environment.")
    main()
