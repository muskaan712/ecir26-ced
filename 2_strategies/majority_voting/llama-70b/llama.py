#!/usr/bin/env python3
# Llama-3.3-70B (GGUF via vLLM OpenAI API) — CED with 5 ERR + 3 NOT few-shots (from TRAIN) + Majority Voting
# - SAME prompt text as your 8B script (uncertainty → NOT)
# - Few-shot: sample 5×ERR + 3×NOT from TRAIN (w/ replacement if needed), reorder to END ON NOT
# - Proper Llama chat formatting (user→assistant shots)
# - Deterministic by default; optional majority voting via OpenAI `n` (temperature>0)
# - Strict label sanitization

import re
import requests
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

# ===================== CONFIG =====================
VLLM_BASE_URL   = "http://127.0.0.1:8000"
MODEL_ID        = "llama33-70b-q4km"   # must match --served-model-name

DEV_TSV         = "/home/ni124545/llm/data/wmt21/ende_majority_dev.tsv"
TRAIN_TSV       = "/home/ni124545/llm/data/wmt21/ende_majority_train.tsv"

EVAL_LIMIT      = 1000                 # set 0/None for all

# Decoding (label-only)
TIMEOUT_SEC     = 180
MAX_NEW_TOKENS  = 3
TEMPERATURE     = 0.0                  # used when USE_MAJORITY_VOTE=False
TOP_P           = 1.0
STOP_SEQ        = ["\n"]

# Few-shot sampling (match 8B code)
USE_FEW_SHOT        = True
FEW_SHOT_ERR_CNT    = 5
FEW_SHOT_NOT_CNT    = 3
RANDOM_STATE        = 42

# Majority voting
USE_MAJORITY_VOTE = True      # set False for single deterministic prediction
N_VOTES           = 3         # odd number (3 or 5 typical)
TEMP_FOR_VOTE     = 0.2       # slight randomness for diverse votes
TIE_BREAK         = "NOT"     # choose on tie (NOT = specificity-leaning)
VOTE_DEBUG_PRINT  = False     # print raw votes per row

assert not USE_MAJORITY_VOTE or (N_VOTES >= 1 and N_VOTES % 2 == 1), "N_VOTES must be odd >= 1"

# Tie-break / default (match the 8B prompt's uncertainty rule)
DEFAULT_LABEL   = "NOT"

API_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

# ===================== PROMPT (same as 8B) =====================
INSTRUCTION_HEADER = (
     "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
    "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
    "or only has minor imperfections.\n"
    "Do not add any explanation, punctuation, or additional text."
)

# ===================== HELPERS =====================
def load_tsv_noheader(path: str) -> pd.DataFrame:
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

def sample_few_shot_examples(train_tsv: str,
                             n_err: int,
                             n_not: int,
                             random_state: int = 42):
    """
    Sample 5 ERR + 3 NOT from TRAIN (with replacement if needed).
    Return list of dicts {src, mt, label} (uppercased label).
    """
    train_df = load_tsv_noheader(train_tsv)
    train_df = train_df[train_df["label"].isin(["ERR","NOT"])]
    def _sample(df, k):
        if len(df) == 0: return df
        if len(df) >= k: return df.sample(k, random_state=random_state)
        return df.sample(k, replace=True, random_state=random_state)
    err_df = _sample(train_df[train_df["label"] == "ERR"], n_err)
    not_df = _sample(train_df[train_df["label"] == "NOT"], n_not)

    ex = []
    for _, r in err_df.iterrows():
        ex.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "ERR"})
    for _, r in not_df.iterrows():
        ex.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "NOT"})
    return ex

def reorder_end_on_not(examples):
    """
    Reorder few-shots to reduce ERR bias: interleave and ensure LAST = NOT.
    """
    if not examples:
        return examples
    errs = [e for e in examples if e["label"] == "ERR"]
    nots = [e for e in examples if e["label"] == "NOT"]
    mixed = []
    while errs or nots:
        if nots:
            mixed.append(nots.pop(0))
        if errs:
            mixed.append(errs.pop(0))
    if mixed and mixed[-1]["label"] != "NOT":
        # move a NOT to the end
        for i in range(len(mixed)-1, -1, -1):
            if mixed[i]["label"] == "NOT":
                mixed.append(mixed.pop(i))
                break
    return mixed

def build_messages_llama(src: str, mt: str, few_shots=None):
    """
    Llama-3 instruct formatting:
      system: instructions only
      few-shots: user(Q with 'Label (ERR or NOT):') → assistant(A label)
      final: user with current example
    """
    msgs = [{"role": "system", "content": INSTRUCTION_HEADER}]
    if USE_FEW_SHOT and few_shots:
        for ex in few_shots:
            q = f"Source (EN): {ex['src']}\nMT (DE): {ex['mt']}\nLabel (ERR or NOT):"
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": ex["label"]})
    user_q = f"Source (EN): {src.strip()}\nMT (DE): {mt.strip()}\nLabel (ERR or NOT):"
    msgs.append({"role": "user", "content": user_q})
    return msgs

def sanitize_label(text: str) -> str:
    if not text:
        return DEFAULT_LABEL
    t = (text or "").strip().upper()
    if t in ("ERR","NOT"):
        return t
    # salvage
    t2 = re.sub(r"[`\"'“”‘’]", " ", t)
    err_pos = next((m.start() for m in re.finditer(r"\bERR\b", t2)), None)
    not_pos = next((m.start() for m in re.finditer(r"\bNOT\b", t2)), None)
    if err_pos is not None and not_pos is not None:
        return "ERR" if err_pos < not_pos else "NOT"
    if err_pos is not None: return "ERR"
    if not_pos is not None: return "NOT"
    return DEFAULT_LABEL

def call_llm_single(messages):
    """Single deterministic call (n=1)."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stop": STOP_SEQ,
    }
    r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_llm_multi(messages, n, temperature, top_p=1.0):
    """Request n samples in one call; return list of assistant contents."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": STOP_SEQ,
    }
    r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    outs = []
    for ch in data.get("choices", []):
        msg = ch.get("message", {})
        outs.append((msg.get("content") or "").strip())
    return outs

# ===================== MAIN =====================
def main():
    # load data
    df_full = load_tsv_noheader(DEV_TSV)
    eval_df = df_full.head(EVAL_LIMIT) if EVAL_LIMIT else df_full
    rows = list(eval_df.itertuples(index=False))

    # build few-shots ONCE from TRAIN, reorder to end on NOT
    few_shots = None
    if USE_FEW_SHOT:
        few_shots = sample_few_shot_examples(
            TRAIN_TSV, FEW_SHOT_ERR_CNT, FEW_SHOT_NOT_CNT, RANDOM_STATE
        )
        few_shots = reorder_end_on_not(few_shots)

    y_true, y_pred = [], []
    for row in tqdm(rows, total=len(rows), desc=f"Evaluating first {len(rows)} rows"):
        msgs = build_messages_llama(row.src, row.mt, few_shots)
        try:
            if USE_MAJORITY_VOTE and N_VOTES > 1:
                samples = call_llm_multi(msgs, n=N_VOTES, temperature=TEMP_FOR_VOTE, top_p=TOP_P)
                labels = [sanitize_label(s) for s in samples]
                tally = Counter(labels)
                if VOTE_DEBUG_PRINT:
                    print(f"[VOTES] raw={samples} -> labels={labels} -> counts={dict(tally)}")
                if tally["ERR"] > tally["NOT"]:
                    raw_label = "ERR"
                elif tally["NOT"] > tally["ERR"]:
                    raw_label = "NOT"
                else:
                    raw_label = TIE_BREAK.upper()
            else:
                raw = call_llm_single(msgs)
                raw_label = sanitize_label(raw)
        except Exception:
            raw_label = DEFAULT_LABEL

        y_pred.append(raw_label)
        y_true.append(row.label)

    # preview
    for i, (yt, yp) in enumerate(zip(y_true, y_pred), 1):
        if i > 10: break
        print(f"[{i:03d}] TRUE={yt}  PRED={yp}")

    # metrics
    map01 = {"ERR":1,"NOT":0}
    yt = [map01.get(y,0) for y in y_true]
    yp = [map01.get(y,0) for y in y_pred]
    mcc = matthews_corrcoef(yt, yp) if len(yt) > 1 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, labels=[1,0], zero_division=0)
    acc = (pd.Series(yt)==pd.Series(yp)).mean()
    cm  = confusion_matrix(yt, yp, labels=[1,0])
    cm_df = pd.DataFrame(cm, index=["ERR_true","NOT_true"], columns=["ERR_pred","NOT_pred"])

    print(f"\nSubset size: {len(eval_df)}")
    if USE_FEW_SHOT:
        print(f"Few-shot demos: {FEW_SHOT_ERR_CNT} ERR + {FEW_SHOT_NOT_CNT} NOT (from TRAIN, rs={RANDOM_STATE}), reordered end=NOT")
    if USE_MAJORITY_VOTE and N_VOTES > 1:
        print(f"Majority vote: {N_VOTES} samples @ T={TEMP_FOR_VOTE}")
    print(f"MCC: {mcc:.4f}  F1-ERR: {f1[0]:.4f}  F1-NOT: {f1[1]:.4f}  Acc: {acc:.4f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print(cm_df.to_string())

if __name__ == "__main__":
    main()
