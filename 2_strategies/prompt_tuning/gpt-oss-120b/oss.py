#!/usr/bin/env python3
# GPT-OSS 120B CED evaluation via vLLM — long decode + FINAL-channel parsing (like 20B)

import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

# ===================== CONFIG =============================================
VLLM_BASE_URL   = "http://127.0.0.1:8000"
MODEL_ID        = "gpt-oss-120b"

DEV_TSV         = "/home/ni124545/llm/data/wmt21/ende_majority_dev.tsv"
TRAIN_TSV       = "/home/ni124545/llm/data/wmt21/ende_majority_train.tsv"

# Evaluate only first N rows (set None/0 to use ALL)
EVAL_LIMIT      = 1000

# Decoding (match 20B: allow long reasoning and parse FINAL)
TIMEOUT_SEC     = 300
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.0
TOP_P           = 1.0
STOP_TOKENS     = ["<|end|>", "<|return|>"]  # lets the model close the message naturally

# Few-shot (keep your 5 ERR / 3 NOT)
USE_FEW_SHOT        = True
FEW_SHOT_ERR_CNT    = 5
FEW_SHOT_NOT_CNT    = 3
RANDOM_STATE        = 42

# ───────────────────────── Optimized SYSTEM_PROMPT ────────────────────────────
SYSTEM_PROMPT = """
You are a bilingual (EN→DE) translation safety checker for sentence-level Critical Error Detection (CED).

Task
- Compare EN (source) with DE (translation).
- Decide if DE deviates in meaning from EN in a way that could mislead with real-world consequences.
- Think privately; then output a single uppercase label.

How to think (channels)
- Use the <analysis> channel to do a brief checklist.
- In the <final> channel output EXACTLY one token: ERR or NOT (no punctuation, no extra text).

Mark ERR if ≥1 of these is caused by translation error (introduced, deleted, mistranslated, or gibberish):
• TOX — Toxicity (hate/violence/profanity) meaning deviates. If toxicity in EN is faithfully preserved in DE, NOT.
• SAF — Health/safety risk meaning deviates (e.g., warnings removed or unsafe action introduced).
• NAM — Named entity deviation (person/org/place/username added/removed/changed; nonsense transliteration; left untranslated when it should be). Minor variations that still clearly refer to the same entity → NOT.
• SEN — Sentiment/negation flip (negation added/removed; polarity reversed; “possibly”→“with certainty”, etc.).
• NUM — Numbers/units/time/date changed or dropped. Conversions are fine only if numeric value is correctly converted (e.g., 0 °C ↔ 32 °F).

Do NOT mark ERR for non-critical issues when meaning is preserved:
style/awkwardness; locale conventions; minor grammar/typos/casing/punctuation; small paraphrases (e.g., “my page”→“the page”).

Decision rule
- Choose ERR only if a specific token/phrase/number/entity in DE causes a clear deviation in {TOX, SAF, NAM, SEN, NUM}.
- Otherwise choose NOT. When uncertain, prefer NOT.

Answer format
<analysis>
Checklist: TOX? SAF? NAM? SEN? NUM? Note any triggers if present.
</analysis>
<final>
ERR
</final>
"""

API_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

# ===================== Helpers =============================================
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

def build_messages_zero_shot(src: str, mt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}\n\nProvide your FINAL decision."},
    ]

def build_messages_fewshot(examples, src: str, mt: str):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user",      "content": f"EN: {ex['src'].strip()}\nDE: {ex['mt'].strip()}\n\nProvide your FINAL decision."})
        # Assistant replies with label only (acts as FINAL)
        msgs.append({"role": "assistant", "content": ex['label'].strip().upper()})
    msgs.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}\n\nProvide your FINAL decision."})
    return msgs

def select_few_shot_examples_from_train(train_tsv: str, n_err: int, n_not: int, random_state: int = 42):
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
    # Combine all potential fields that may contain model text.
    msg = choice.get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    alt = choice.get("text") or ""
    return (" ".join([content, reasoning, alt])).strip()

def extract_final_or_label(text: str) -> str:
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
    # Conservative fallback to ERR
    s = (t or "").strip().upper()
    if "ERR" in s and "NOT" in s:
        return "ERR" if s.index("ERR") < s.index("NOT") else "NOT"
    if "ERR" in s: return "ERR"
    if "NOT" in s: return "NOT"
    return "ERR"

# --- Inference (unguided, long decode, FINAL parsing) ------------------------
def infer_one_with_retry(src, mt, examples=None, max_retries=3, backoff=1.25):
    messages = build_messages_fewshot(examples, src, mt) if examples else build_messages_zero_shot(src, mt)

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stop": STOP_TOKENS,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SEC)

            # Retry on rate limits / transient server errors
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    time.sleep(backoff ** attempt)
                    continue
                else:
                    return "ERR"  # conservative on hard failure

            r.raise_for_status()
            resp = r.json()
            choice = resp["choices"][0]
            raw = extract_text_from_choice(choice)
            parsed = extract_final_or_label(raw)
            return sanitize_label(parsed)

        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                return "ERR"
            time.sleep(backoff ** attempt)

    return "ERR"

# ===================== Main ================================================
def main():
    """Run inference over the dataset and report metrics."""
    df_full = load_tsv_noheader(DEV_TSV)
    eval_df = df_full.head(EVAL_LIMIT) if EVAL_LIMIT else df_full
    rows = list(eval_df.itertuples(index=False))

    print(f"Processing: {len(rows)} rows | Few-shot: {'ON' if USE_FEW_SHOT else 'OFF'} | Max tokens: {MAX_NEW_TOKENS}")

    few_shots = None
    if USE_FEW_SHOT:
        few_shots = select_few_shot_examples_from_train(
            TRAIN_TSV, FEW_SHOT_ERR_CNT, FEW_SHOT_NOT_CNT, RANDOM_STATE
        )

    y_true, y_pred = [], []
    preview_k = min(10, len(rows))
    print(f"\n=== PREVIEW (first {preview_k}) ===")

    for i, row in enumerate(tqdm(rows, desc="Evaluating", unit="row"), 1):
        pred = infer_one_with_retry(row.src, row.mt, examples=few_shots)
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

    print(f"\nProcessed: {len(rows)} rows | Few-shot: {'ON' if USE_FEW_SHOT else 'OFF'}")
    print(f"MCC   : {mcc:.4f}")
    print(f"F1-ERR: {f_err:.4f} | F1-NOT: {f_not:.4f} | Acc: {acc:.4f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR  {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"NOT  {cm[1,0]:5d} {cm[1,1]:5d}")

if __name__ == "__main__":
    main()
