#!/usr/bin/env python3
# zero_shot_gpt4o_inference_3col_errnot.py
#
# Strict 3+ column TSV (no header):
#   col0 = EN (src), col1 = DE (mt), col_last = label ("ERR" or "NOT")
# - Only accepts ERR/NOT labels for evaluation (no BAD/OK mapping).
# - ZERO-SHOT: no few-shot examples are used.
# - Logs class counts for DEV and prints metrics (MCC, F1 per class, CM).

import os, sys, time, logging
from typing import List
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
import openai

# ── Config ─────────────────────────────────────────────────────────────────────
DEV_TSV    = "/home/s13mchop/LLMs/data/wmt22/ende_wmt22_dev.tsv"

# Prefer OPENAI_API_KEY; falls back to OPENAI for your current env
openai.api_key = os.environ.get("OPENAI") or os.environ.get("OPENAI") or ""

MODEL       = "gpt-4o"
MAX_TOKENS  = 3
TEMPERATURE = 0.0
COST_PER_1K = 0.002   # adjust if your actual pricing differs

ALLOWED = {"ERR", "NOT"}

# Optional: limit rows for quick smoke tests; set to 0 or None for all
EVAL_LIMIT = 0

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="inference_gpt4o_zero_shot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)

# ── IO ─────────────────────────────────────────────────────────────────────────
def load_3col_tsv(path: str, tag: str) -> pd.DataFrame:
    """
    Strict 3+ col (no header) loader with tab separator:
    - first col -> src
    - second col -> mt
    - last col -> label (must be ERR/NOT after upper/strip)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        engine="python",
        on_bad_lines="skip",
        dtype=str,
        quoting=3,  # QUOTE_NONE (inch marks like 1/8" won't break parsing)
    )
    if df.shape[1] < 3:
        raise RuntimeError(f"[{tag}] Expected ≥3 columns (EN, DE, LABEL). Got shape={df.shape}. Check the delimiter.")

    src  = df.iloc[:, 0].astype(str)
    mt   = df.iloc[:, 1].astype(str)
    lbl  = df.iloc[:, -1].astype(str)

    out = pd.DataFrame({"src": src, "mt": mt, "label": lbl})
    out["label"] = out["label"].str.strip().str.upper()
    before = len(out)
    out = out[out["label"].isin(ALLOWED)].copy()
    dropped = before - len(out)
    if dropped > 0:
        logging.warning(f"[{tag}] Dropped {dropped} rows with non-{{'ERR','NOT'}} labels.")
    if len(out) == 0:
        raise RuntimeError(f"[{tag}] No usable rows after filtering to ERR/NOT.")
    out["label_id"] = out["label"].map({"ERR":1,"NOT":0})

    c_err = int((out["label"]=="ERR").sum())
    c_not = int((out["label"]=="NOT").sum())
    logging.info(f"[{tag}] rows={len(out)} | ERR={c_err} | NOT={c_not}")
    return out

# ── Prompting (ZERO-SHOT) ─────────────────────────────────────────────────────
def build_messages_zero_shot(src: str, mt: str):
    system_prompt = (
        "You are a precise translation evaluator.\n"
        "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
        "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
        "or only has minor imperfections.\n"
        "Output must be exactly 'ERR' or 'NOT' with no punctuation or explanation."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}"}
    ]
    return messages

def parse_label(text: str) -> str:
    """
    Convert raw model output to 'ERR' or 'NOT'.
    Robust to trailing punctuation or extra words.
    """
    if text is None:
        return "NOT"
    s = str(text).strip().upper()
    # Fast paths
    if s == "ERR": return "ERR"
    if s == "NOT": return "NOT"
    # Containment (avoid cases like "NOT ERR")
    if "ERR" in s and "NOT" not in s: return "ERR"
    if "NOT" in s and "ERR" not in s: return "NOT"
    # First token fallback
    tok = s.split()[0]
    return tok if tok in ALLOWED else "NOT"

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    dev_df = load_3col_tsv(DEV_TSV, "DEV")
    if EVAL_LIMIT and EVAL_LIMIT > 0:
        dev_df = dev_df.head(EVAL_LIMIT).copy()
        logging.info(f"[DEV] Limiting evaluation to first {len(dev_df)} rows.")

    gen_labels: List[str] = []
    preds: List[int] = []
    total_tokens = 0
    t0 = time.time()

    for i, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="GPT-4o zero-shot inference"):
        messages = build_messages_zero_shot(row["src"], row["mt"])
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=1
        )
        raw = resp.choices[0].message.content
        lab = parse_label(raw)
        gen_labels.append(lab)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            total_tokens += usage.prompt_tokens + usage.completion_tokens
            logging.info(
                f"Row {i}: gen={lab!r} raw={raw!r} | prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
            )
        preds.append(1 if lab == "ERR" else 0)

    elapsed = time.time() - t0
    logging.info(f"Total inference time: {elapsed:.2f}s")
    logging.info(f"Total tokens used: {total_tokens}")
    logging.info(f"Estimated cost: ${ (total_tokens/1000.0)*COST_PER_1K :.4f}")

    # Preview
    print("\nFirst 10 results (Generated | True | Pred):\n")
    show_n = min(10, len(dev_df))
    for i in range(show_n):
        gen = gen_labels[i] or "<EMPTY>"
        true = "ERR" if dev_df.iloc[i]["label_id"] == 1 else "NOT"
        pred = "ERR" if preds[i] == 1 else "NOT"
        print(f"#{i+1:2d} {gen!r} | {true} | {pred}")

    labels = dev_df["label_id"].tolist()
    mcc = matthews_corrcoef(labels, preds)
    prf = precision_recall_fscore_support(labels, preds, labels=[1,0], zero_division=0)
    cm  = confusion_matrix(labels, preds, labels=[1,0])

    print(f"\nMCC   : {mcc:.4f}")
    print(f"F1-ERR: {prf[2][0]:.4f}")
    print(f"F1-NOT: {prf[2][1]:.4f}\n")
    print("Confusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"NOT {cm[1,0]:6d} {cm[1,1]:6d}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        raise
