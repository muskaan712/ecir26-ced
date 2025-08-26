#!/usr/bin/env python3
# Zero-shot CED (EN→DE) via vLLM (GGUF server)
# Runs only the first N rows for quick debugging.

import requests, pandas as pd
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

# ===================== CONFIG (edit here) =====================
VLLM_BASE_URL   = "http://127.0.0.1:8000"
MODEL_ID        = "llama33-70b-q4km"   # must match --served-model-name
DEV_TSV         = "/home/ni124545/llm/data/wmt21/ende_majority_dev.tsv"

# Debugging limit
EVAL_LIMIT      = 1000                   # <-- run only first N rows

# Decoding (deterministic baseline)
TIMEOUT_SEC     = 180
MAX_NEW_TOKENS  = 3
TEMPERATURE     = 0.0
TOP_P           = 1.0
TOP_K           = 0  # not sent (server was ignoring extra_body); left here for reference
# Prompt (your content)
SYSTEM_PROMPT = (
"You are a precise translation evaluator.\n"
"Given an English sentence (EN) and its German translation (DE), respond with exactly one token: 'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate or only has minor imperfections.\n"
"Do not add any explanation, punctuation, or additional text."

)
def build_messages(src: str, mt: str):
    """Construct chat messages for classification tasks."""
    user_prompt = f"EN: {src.strip()}\nDE: {mt.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
# =============================================================

API_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

def load_tsv_noheader(path):
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
    df["label"] = df["label"].str.strip().str.upper()
    return df[["src","mt","label"]]

def sanitize_label(text: str) -> str:
    t = text.strip().upper()
    if "ERR" in t and "NOT" in t:
        return "ERR" if t.index("ERR") < t.index("NOT") else "NOT"
    if "ERR" in t: return "ERR"
    if "NOT" in t: return "NOT"
    return "ERR"

def infer_one(src, mt):
    payload = {
        "model": MODEL_ID,
        "messages": build_messages(src, mt),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        # Note: not sending extra_body; server logged it as ignored.
        # If your vLLM build supports top_k directly, add: "top_k": TOP_K
    }
    r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    return sanitize_label(out)

def main():
    """Run inference over the dataset and report metrics."""
    df_full = load_tsv_noheader(DEV_TSV)
    df = df_full.head(EVAL_LIMIT) if EVAL_LIMIT else df_full

    y_true, y_pred = [], []
    rows = list(df.itertuples(index=False))
    for row in tqdm(rows, total=len(rows), desc=f"Evaluating first {len(rows)} rows"):
        y_pred.append(infer_one(row.src, row.mt))
        y_true.append(row.label)

    # quick per-row print for debugging
    for i, (yt, yp) in enumerate(zip(y_true, y_pred), 1):
        print(f"[{i:03d}] TRUE={yt}  PRED={yp}")

    # metrics on the subset
    map01 = {"ERR":1,"NOT":0}
    yt = [map01.get(y,0) for y in y_true]
    yp = [map01.get(y,0) for y in y_pred]

    mcc = matthews_corrcoef(yt, yp) if len(yt) > 1 else 0.0

    # FIX: precision_recall_fscore_support returns 4 arrays; index F1 for each label
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=[1,0], zero_division=0)
    f_err, f_not = f1[0], f1[1]

    acc = (pd.Series(yt)==pd.Series(yp)).mean()
    cm  = confusion_matrix(yt, yp, labels=[1,0])
    cm_df = pd.DataFrame(cm, index=["ERR_true","NOT_true"], columns=["ERR_pred","NOT_pred"])

    print(f"\nSubset size: {len(df)}")
    print(f"MCC: {mcc:.4f}  F1-ERR: {f_err:.4f}  F1-NOT: {f_not:.4f}  Acc: {acc:.4f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print(cm_df.to_string())

if __name__ == "__main__":
    main()
