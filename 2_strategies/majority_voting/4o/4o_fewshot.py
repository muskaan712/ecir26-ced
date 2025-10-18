#!/usr/bin/env python3
"""Run GPT-4o few-shot inference with majority voting for critical error detection.

The script reproduces the original notebook workflow for the WMT'21 Task 3
evaluation, but adds lightweight logging and safer configuration so it can be
reused by other practitioners.  Input data is expected to be a headerless TSV
with at least three columns where the first two correspond to the English
source (``src``) and German machine translation (``mt``), and the final column
contains an ``ERR``/``NOT`` label.  The committee uses three samples drawn from
few-shot exemplars with temperature ``0.2`` and falls back to ``NOT`` on ties.
"""

import os
import sys
import time
import logging
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
import openai
from collections import Counter

# ── Config ─────────────────────────────────────────────────────────────────────
DEV_TSV = os.environ.get("DEV_TSV", "/path/to/dev_dataset.tsv")
TRAIN_TSV = os.environ.get("TRAIN_TSV", "/path/to/train_dataset.tsv")

# Prefer OPENAI_API_KEY; falls back to OPENAI for your current env
openai.api_key = os.environ.get("OAPI") or os.environ.get("OPENAI") or ""

MODEL         = "gpt-4o"
MAX_TOKENS    = 3
TEMPERATURE   = 0.2          # ← set as requested
COST_PER_1K   = 0.002        # adjust if you want exact pricing for your model
COMMITTEE_K   = 3            # ← majority vote committee size
TIE_FALLBACK  = "NOT"        # deterministic tie-breaker

FEW_SHOT_ERR_CNT = 5
FEW_SHOT_NOT_CNT = 3

ALLOWED = {"ERR", "NOT"}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="inference_gpt4o_few_shot_mv.log",
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

# ── Few-shot selection ─────────────────────────────────────────────────────────
def sample_with_replace(df: pd.DataFrame, label: str, k: int, seed: int = 42) -> pd.DataFrame:
    """Return ``k`` samples for ``label`` (with replacement when necessary)."""
    sub = df[df["label"] == label]
    n = len(sub)
    if n == 0:
        raise RuntimeError(f"No rows for class '{label}' in TRAIN after filtering to ERR/NOT.")
    replace = n < k
    if replace:
        logging.warning(f"Class '{label}' has {n} rows; sampling {k} with replace=True.")
    return sub.sample(k, random_state=seed, replace=replace)

def select_few_shot_examples(train_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Prepare the few-shot exemplar set consumed by the majority vote calls."""
    err_examples = sample_with_replace(train_df, "ERR", FEW_SHOT_ERR_CNT)
    not_examples = sample_with_replace(train_df, "NOT", FEW_SHOT_NOT_CNT)
    examples: List[Dict[str, str]] = []
    for _, r in err_examples.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "ERR"})
    for _, r in not_examples.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "NOT"})
    logging.info(f"Prepared few-shot set: ERR={len(err_examples)} | NOT={len(not_examples)}")
    return examples

# ── Prompting ──────────────────────────────────────────────────────────────────
def build_messages(src: str, mt: str, examples: List[Dict[str, str]]):
    """Construct the chat message list combining the prompt and exemplars."""
    system_prompt = (
        "You are a STRICT binary classifier for WMT’21 Task 3 (Critical Error Detection, EN→DE).\n\n"
        "Goal\n"
        "- Decide if the German MT contains at least one CRITICAL meaning error relative to the English source.\n"
        "- Output EXACTLY one token: ERR or NOT (UPPERCASE, no punctuation, no spaces, no explanation).\n\n"
        "Critical errors (any ⇒ ERR)\n"
        "- TOX: toxicity/hate/violence/profanity introduced, deleted, mistranslated, or left untranslated in a way that changes meaning.\n"
        "- SAF: health/safety risk introduced, deleted, mistranslated, or left untranslated (e.g., advice flips, risky omissions).\n"
        "- NAM: named entity added/removed/mistranslated/gibberish/unrecoverable transliteration (people/org/place/product/username).\n"
        "- SEN: sentiment polarity or negation flipped or materially strengthened/weakened (e.g., “don’t”→“do”, “possibly”→“certainly”).\n"
        "- NUM: wrong/missing/added numbers, dates, times, units that change meaning (e.g., 8am↔8pm, km↔miles without conversion).\n\n"
        "Non-critical (ignore; still ⇒ NOT)\n"
        "- Style/register/awkwardness/locale punctuation.\n"
        "- Fluency/grammar/typos that don’t change critical meaning.\n"
        "- Minor lexical changes that keep meaning (page↔site, small intensifier changes that don’t flip sentiment).\n"
        "- Correct transfer of toxicity that was already in the source (not an error).\n\n"
        "Decision policy (optimize reliability/MCC)\n"
        "- Mark ERR only with CLEAR evidence of a critical deviation in the categories above.\n"
        "- If uncertain, default to NOT.\n\n"
        "Procedure (think silently; do not write rationale)\n"
        "1) Read EN source and DE MT. If helpful, internally paraphrase MT to EN.\n"
        "2) Compare meanings with attention to TOX/SAF/NAM/SEN/NUM.\n"
        "3) Decide. Output ONLY: ERR or NOT.\n\n"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for ex in examples:
        messages.append({"role": "user", "content": f"EN: {ex['src']}\nDE: {ex['mt']}"})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}"})
    return messages

def parse_label(text: str) -> str:
    """
    Convert raw model output to 'ERR' or 'NOT'.
    Robust to trailing punctuation or extra words.
    """
    if text is None:
        return "NOT"
    s = str(text).strip().upper()
    if s == "ERR": return "ERR"
    if s == "NOT": return "NOT"
    if "ERR" in s and "NOT" not in s: return "ERR"
    if "NOT" in s and "ERR" not in s: return "NOT"
    tok = s.split()[0]
    return tok if tok in ALLOWED else "NOT"

def majority_vote(labels: List[str], tie_fallback: str = TIE_FALLBACK) -> str:
    """
    Choose the modal label; deterministic tie fallback.
    """
    cnt = Counter(labels)
    if not cnt:
        return tie_fallback
    top = cnt.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    # tie
    return tie_fallback

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Execute inference across the dev TSV and report aggregate metrics."""
    dev_df   = load_3col_tsv(DEV_TSV,   "DEV")
    train_df = load_3col_tsv(TRAIN_TSV, "TRAIN")
    examples = select_few_shot_examples(train_df)

    gen_labels: List[str] = []
    preds: List[int] = []
    total_tokens = 0
    t0 = time.time()

    for i, row in tqdm(dev_df.iterrows(), total=len(dev_df),
                       desc=f"GPT-4o few-shot inference (K={COMMITTEE_K}, T={TEMPERATURE})"):
        messages = build_messages(row["src"], row["mt"], examples)
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=COMMITTEE_K
        )

        votes: List[str] = []
        for j, ch in enumerate(resp.choices):
            raw = getattr(ch.message, "content", None)
            lab = parse_label(raw)
            votes.append(lab)

        final_label = majority_vote(votes, TIE_FALLBACK)
        gen_labels.append(final_label)
        preds.append(1 if final_label == "ERR" else 0)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            # For n>1, OpenAI usage aggregates across choices
            total_tokens += usage.prompt_tokens + usage.completion_tokens
            logging.info(
                f"Row {i}: votes={votes} -> final={final_label} | "
                f"prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
            )
        else:
            logging.info(f"Row {i}: votes={votes} -> final={final_label} | usage=N/A")

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
