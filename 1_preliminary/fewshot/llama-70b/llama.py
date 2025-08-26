#!/usr/bin/env python3
# Zero-shot / Few-shot CED (EN→DE) via vLLM (GGUF server)
# Runs only the first N rows for quick debugging.

import requests, pandas as pd
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

# ===================== CONFIG (edit here) =====================
VLLM_BASE_URL   = "http://127.0.0.1:8000"
MODEL_ID        = "llama33-70b-q4km"   # must match --served-model-name
DEV_TSV         = "/home/ni124545/llm/data/wmt21/ende_majority_dev.tsv"
TRAIN_TSV       = "/home/ni124545/llm/data/wmt21/ende_majority_train.tsv"

# Debugging / eval slice
EVAL_LIMIT      = 1000                 # <-- evaluate only first N rows (set 0/None for all)

# Decoding (deterministic baseline)
TIMEOUT_SEC     = 180
MAX_NEW_TOKENS  = 3
TEMPERATURE     = 0.0
TOP_P           = 1.0
TOP_K           = 0   # not sent (server ignores extra_body); left here for reference

# Few-shot config (GPT-4o logic: 5 ERR + 3 NOT from TRAIN, random_state=42)
USE_FEW_SHOT        = True            # <-- set True to enable few-shot prompting
FEW_SHOT_ERR_CNT    = 5
FEW_SHOT_NOT_CNT    = 3
RANDOM_STATE        = 42

# Prompt
SYSTEM_PROMPT = (
    "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
    "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
    "or only has minor imperfections.\n"
    "Do not add any explanation, punctuation, or additional text."
)

def build_messages(src: str, mt: str):
    """Zero-shot messages."""
    user_prompt = f"EN: {src.strip()}\nDE: {mt.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

def build_messages_fewshot(examples, src: str, mt: str):
    """
    Few-shot messages using conversational demonstrations:
      user: example pair
      assistant: gold label ('ERR' / 'NOT')
    Order: all ERR examples first, then NOT (to match your GPT-4o script).
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user",      "content": f"EN: {ex['src'].strip()}\nDE: {ex['mt'].strip()}"})
        msgs.append({"role": "assistant", "content": ex['label'].strip().upper()})
    msgs.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}"})
    return msgs
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

def select_few_shot_examples_from_train(train_tsv: str,
                                        n_err: int,
                                        n_not: int,
                                        random_state: int = 42):
    """
    Replicates the GPT-4o script's logic:
      - Load TRAIN_TSV
      - Sample 5 ERR (oversample) + 3 NOT with random_state for reproducibility
      - Return list of {src, mt, label}; ERRs come first, then NOTs
    Robust if the split is smaller than requested (falls back to sampling with replacement).
    """
    train_df = load_tsv_noheader(train_tsv)

    # Ensure we have uppercase labels and only valid classes
    train_df = train_df[train_df["label"].isin(["ERR","NOT"])]

    def _sample(df, k):
        if len(df) == 0:
            return df
        if len(df) >= k:
            return df.sample(k, random_state=random_state)
        # Not enough rows -> sample with replacement to reach k
        return df.sample(k, replace=True, random_state=random_state)

    err_df = _sample(train_df[train_df["label"] == "ERR"], n_err)
    not_df = _sample(train_df[train_df["label"] == "NOT"], n_not)

    examples = []
    # ERR first (to mirror your few_shot_gpt4o_inference.py order)
    for _, r in err_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "ERR"})
    # then NOT
    for _, r in not_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "NOT"})

    return examples

def infer_one(src, mt, examples=None):
    """If examples is provided, do few-shot; otherwise zero-shot."""
    messages = build_messages_fewshot(examples, src, mt) if examples else build_messages(src, mt)
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        # If your vLLM build supports top_k directly in body, you can add: "top_k": TOP_K
    }
    r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    return sanitize_label(out)

def main():
    """Run inference over the dataset and report metrics."""
    df_full = load_tsv_noheader(DEV_TSV)
    eval_df = df_full.head(EVAL_LIMIT) if EVAL_LIMIT else df_full
    rows = list(eval_df.itertuples(index=False))

    # Build few-shot examples ONCE from TRAIN (no leakage)
    few_shots = None
    if USE_FEW_SHOT:
        few_shots = select_few_shot_examples_from_train(
            TRAIN_TSV, FEW_SHOT_ERR_CNT, FEW_SHOT_NOT_CNT, RANDOM_STATE
        )

    y_true, y_pred = [], []
    for row in tqdm(rows, total=len(rows), desc=f"Evaluating first {len(rows)} rows"):
        y_pred.append(infer_one(row.src, row.mt, examples=few_shots))
        y_true.append(row.label)

    # quick per-row print for debugging
    for i, (yt, yp) in enumerate(zip(y_true, y_pred), 1):
        print(f"[{i:03d}] TRUE={yt}  PRED={yp}")

    # metrics on the subset
    map01 = {"ERR":1,"NOT":0}
    yt = [map01.get(y,0) for y in y_true]
    yp = [map01.get(y,0) for y in y_pred]

    mcc = matthews_corrcoef(yt, yp) if len(yt) > 1 else 0.0
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=[1,0], zero_division=0)
    f_err, f_not = f1[0], f1[1]

    acc = (pd.Series(yt)==pd.Series(yp)).mean()
    cm  = confusion_matrix(yt, yp, labels=[1,0])
    cm_df = pd.DataFrame(cm, index=["ERR_true","NOT_true"], columns=["ERR_pred","NOT_pred"])

    print(f"\nSubset size: {len(eval_df)}")
    if USE_FEW_SHOT:
        print(f"Few-shot demos: {FEW_SHOT_ERR_CNT} ERR + {FEW_SHOT_NOT_CNT} NOT (from TRAIN, rs={RANDOM_STATE})")
    print(f"MCC: {mcc:.4f}  F1-ERR: {f_err:.4f}  F1-NOT: {f_not:.4f}  Acc: {acc:.4f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print(cm_df.to_string())

if __name__ == "__main__":
    main()
