#!/usr/bin/env python3
"""Evaluate GPT-OSS 20B for critical error detection with vLLM majority voting."""

import os
import re
from collections import Counter

import pandas as pd
import requests
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from tqdm import tqdm

# ===================== CONFIG =============================================
VLLM_BASE_URL   = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL_ID        = os.environ.get("MODEL_ID", "gpt-oss-120b")

DEV_TSV = os.environ.get("DEV_TSV", "/path/to/dev_dataset.tsv")
TRAIN_TSV = os.environ.get("TRAIN_TSV", "/path/to/train_dataset.tsv")

# Decoding (allow long reasoning and parse FINAL)
TIMEOUT_SEC     = int(os.environ.get("TIMEOUT_SEC", "300"))
MAX_NEW_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TEMPERATURE     = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P           = float(os.environ.get("TOP_P", "1.0"))
STOP_TOKENS     = os.environ.get("STOP_TOKENS", "<|end|>,<|return|>").split(",")

# Few-shot configuration (static)
USE_FEW_SHOT     = True          # ← set to False for zero-shot
FEW_SHOT_ERR_CNT = 5
FEW_SHOT_NOT_CNT = 3
RANDOM_STATE     = 42

# Majority voting
USE_MAJORITY_VOTE = True    # ← set False for single long decode
N_VOTES           = 3       # odd (3/5 typical)
TEMP_FOR_VOTE     = 0.2     # mild diversity
TOP_P_FOR_VOTE    = 1.0
TIE_BREAK         = "NOT"   # tie → NOT (specificity-leaning)
VOTE_DEBUG_PRINT  = False   # print raw vote labels for first rows

assert not USE_MAJORITY_VOTE or (N_VOTES >= 1 and N_VOTES % 2 == 1), "N_VOTES must be odd >= 1"
DEFAULT_LABEL = TIE_BREAK.upper()

SYSTEM_PROMPT = (
    "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), think carefully and then provide a FINAL decision.\n"
    "Decision rules:\n"
    "- ERR: major meaning error, omission, or inaccuracy.\n"
    "- NOT: accurate or only minor imperfections.\n"
    "In your FINAL message, output ONLY one token: ERR or NOT.\n"
    "Use your usual internal structure; if you use channels, ensure the FINAL channel contains exactly ERR or NOT."
)

API_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

# ===================== Helpers =============================================
def load_tsv_noheader(path: str) -> pd.DataFrame:
    """Load TSV data without headers and standardize the columns."""
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

def build_messages_zero_shot(src: str, mt: str):
    """Build a zero-shot prompt for a single sentence pair."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}\n\nProvide your FINAL decision."},
    ]

def build_messages_fewshot(examples, src: str, mt: str):
    """Construct a message list including few-shot exemplars."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user",      "content": f"EN: {ex['src'].strip()}\nDE: {ex['mt'].strip()}\n\nProvide your FINAL decision."})
        msgs.append({"role": "assistant", "content": ex['label'].strip().upper()})
    msgs.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}\n\nProvide your FINAL decision."})
    return msgs

def select_few_shot_examples_from_train(
    train_tsv: str,
    n_err: int,
    n_not: int,
    random_state: int = 42,
):
    """Sample ERR/NOT exemplars from the training file for few-shot prompting."""
    df = load_tsv_noheader(train_tsv)
    df = df[df["label"].isin(["ERR","NOT"])]

    def _sample(d, k):
        if len(d) == 0: return d
        if len(d) >= k: return d.sample(k, random_state=random_state)
        return d.sample(k, replace=True, random_state=random_state)

    err_df = _sample(df[df["label"] == "ERR"], n_err)
    not_df = _sample(df[df["label"] == "NOT"], n_not)

    examples = []
    for _, r in err_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "ERR"})
    for _, r in not_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "NOT"})

    print(f"Few-shot demos prepared: {len(err_df)} ERR + {len(not_df)} NOT")
    return examples

# --- Parsing like 20B: prefer FINAL channel, else first ERR/NOT --------------
LABEL_RE = re.compile(r"\b(ERR|NOT)\b", re.I)
FINAL_BLOCK_RE = re.compile(r"<\|channel\|\>final<\|message\|\>(.*?)(?:<\|end\|\>|<\|return\|\>|$)", re.S)

def extract_text_from_choice(choice: dict) -> str:
    """Combine the vLLM choice fields into one text payload."""
    msg = choice.get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    alt = choice.get("text") or ""
    return (" ".join([content, reasoning, alt])).strip()

def extract_final_or_label(text: str) -> str:
    """Prefer FINAL-channel output; otherwise fall back to the first label token."""
    if not text:
        return ""
    m = FINAL_BLOCK_RE.search(text)
    if m:
        final_text = (m.group(1) or "").strip()
        m2 = LABEL_RE.search(final_text)
        return m2.group(1).upper() if m2 else final_text
    m3 = LABEL_RE.search(text)
    return m3.group(1).upper() if m3 else text.strip()

def sanitize_label(t: str) -> str:
    """Normalize noisy text into the canonical ERR/NOT labels."""
    s = (t or "").strip().upper()
    if "ERR" in s and "NOT" in s:
        return "ERR" if s.index("ERR") < s.index("NOT") else "NOT"
    if "ERR" in s: return "ERR"
    if "NOT" in s: return "NOT"
    return DEFAULT_LABEL

# --- vLLM call helpers -------------------------------------------------------
def _post_vllm(payload, max_retries=3):
    """Issue a POST request to vLLM with retry support."""
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
            if r.status_code >= 500:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            continue

# --- Inference (single) ------------------------------------------------------
def infer_one_with_retry(src, mt, examples=None, max_retries=3):
    """Run a single inference pass and sanitize the resulting label."""
    messages = build_messages_fewshot(examples, src, mt) if examples else build_messages_zero_shot(src, mt)
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stop": STOP_TOKENS,
        # "reasoning": {"effort": "low"},
    }
    try:
        resp = _post_vllm(payload, max_retries=max_retries)
        choice = resp["choices"][0]
        raw = extract_text_from_choice(choice)
        parsed = extract_final_or_label(raw)
        return sanitize_label(parsed)
    except Exception:
        return DEFAULT_LABEL

# --- Inference (majority vote) ----------------------------------------------
def infer_majority_with_retry(
    src,
    mt,
    examples=None,
    n_votes=3,
    temp_for_vote=0.2,
    top_p_for_vote=1.0,
    max_retries=3,
    vote_debug=False,
):
    """Collect ``n_votes`` predictions (batched when possible) and majority vote."""
    messages = build_messages_fewshot(examples, src, mt) if examples else build_messages_zero_shot(src, mt)

    # Try efficient single call with `n`
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": temp_for_vote,
        "top_p": top_p_for_vote,
        "n": n_votes,
        "stop": STOP_TOKENS,
        # "reasoning": {"effort": "low"},
    }
    labels = []
    try:
        resp = _post_vllm(payload, max_retries=max_retries)
        choices = resp.get("choices", [])
        if len(choices) == n_votes:
            for ch in choices:
                raw = extract_text_from_choice(ch)
                parsed = extract_final_or_label(raw)
                labels.append(sanitize_label(parsed))
        else:
            # If server ignored `n`, fall back to sequential
            labels = []
    except Exception:
        labels = []

    # Sequential fallback if needed
    if len(labels) != n_votes:
        labels = []
        for _ in range(n_votes):
            lab = infer_one_with_retry(src, mt, examples=examples, max_retries=max_retries)
            labels.append(lab)

    if vote_debug:
        print(f"[VOTE] labels={labels}")

    tally = Counter(labels)
    if tally["ERR"] > tally["NOT"]:
        return "ERR"
    if tally["NOT"] > tally["ERR"]:
        return "NOT"
    return DEFAULT_LABEL

# ===================== Main ================================================
def main():
    """Evaluate the dataset and print preview rows plus summary metrics."""
    df_full = load_tsv_noheader(DEV_TSV)
    eval_df = df_full
    rows = list(eval_df.itertuples(index=False))

    print(f"Processing: {len(rows)} rows | Few-shot: {'ON' if USE_FEW_SHOT else 'OFF'} | "
          f"Max tokens: {MAX_NEW_TOKENS} | MV: {'ON' if USE_MAJORITY_VOTE and N_VOTES>1 else 'OFF'}")

    few_shots = None
    if USE_FEW_SHOT:
        few_shots = select_few_shot_examples_from_train(
            TRAIN_TSV, FEW_SHOT_ERR_CNT, FEW_SHOT_NOT_CNT, RANDOM_STATE
        )

    y_true, y_pred = [], []
    preview_k = min(10, len(rows))
    print(f"\n=== PREVIEW (first {preview_k}) ===")

    for i, row in enumerate(tqdm(rows, desc="Evaluating", unit="row"), 1):
        if USE_MAJORITY_VOTE and N_VOTES > 1:
            pred = infer_majority_with_retry(
                row.src, row.mt,
                examples=few_shots,
                n_votes=N_VOTES,
                temp_for_vote=TEMP_FOR_VOTE,
                top_p_for_vote=TOP_P_FOR_VOTE,
                max_retries=3,
                vote_debug=(VOTE_DEBUG_PRINT and i <= 5)
            )
        else:
            pred = infer_one_with_retry(row.src, row.mt, examples=few_shots, max_retries=3)

        y_true.append(row.label)
        y_pred.append(pred)

        if i <= preview_k:
            print(f"[{i:03d}] TRUE={row.label} | PRED={pred}")

        if i % 200 == 0:
            acc_partial = (pd.Series(y_true) == pd.Series(y_pred)).mean()
            print(f"Progress: {i}/{len(rows)} | partial_acc={acc_partial:.3f}")

    # Metrics
    map01 = {"ERR": 1, "NOT": 0}
    yt = [map01.get(y, 0) for y in y_true]
    yp = [map01.get(y, 0) for y in y_pred]

    mcc = matthews_corrcoef(yt, yp) if len(yt) > 1 else 0.0
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=[1,0], zero_division=0)
    f_err, f_not = f1[0], f1[1]
    acc = (pd.Series(yt) == pd.Series(yp)).mean()
    cm = confusion_matrix(yt, yp, labels=[1,0])

    print(f"\nProcessed: {len(rows)} rows | Few-shot: {'ON' if USE_FEW_SHOT else 'OFF'} "
          f"| Majority vote: {'ON' if USE_MAJORITY_VOTE and N_VOTES>1 else 'OFF'} (N={N_VOTES})")
    print(f"MCC   : {mcc:.4f}")
    print(f"F1-ERR: {f_err:.4f} | F1-NOT: {f_not:.4f} | Acc: {acc:.4f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR  {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"NOT  {cm[1,0]:5d} {cm[1,1]:5d}")

if __name__ == "__main__":
    main()
