#!/usr/bin/env python3
"""Run GPT-4o mini few-shot inference for WMT critical error detection.

The original script targeted a local developer setup.  This revision keeps the
robust TSV loading and evaluation helpers while documenting the workflow and
allowing configurable dataset paths via environment variables.
"""
import logging
import os
import time
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
import openai

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="inference_gpt4o_few_shot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)

# ── Config ─────────────────────────────────────────────────────────────────────
DEV_TSV = os.environ.get("DEV_TSV", "/path/to/dev_dataset.tsv")
TRAIN_TSV = os.environ.get("TRAIN_TSV", "/path/to/train_dataset.tsv")

openai.api_key   = os.getenv("OAPI") or os.getenv("OPENAI_API_KEY") or ""

MODEL            = "gpt-4o-mini"
MAX_TOKENS       = 3
TEMPERATURE      = 0.0
COST_PER_1K      = 0.002

FEW_SHOT_ERR_CNT = 5
FEW_SHOT_NOT_CNT = 3

LABEL_MAP = {"ERR": "ERR", "NOT": "NOT", "BAD": "ERR", "OK": "NOT"}  # harmless if not present
ALLOWED   = {"ERR", "NOT"}

# ── Loading ────────────────────────────────────────────────────────────────────
def _read_tsv(path: str) -> pd.DataFrame:
    """Read a TSV with no assumptions about header/cols."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,            # WMT files typically no header
        dtype=str,
        engine="python",
        quoting=3,              # QUOTE_NONE
        on_bad_lines="skip",
    )

def _coerce_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
      - 3+ cols: take col0→src, col1→mt, last→label
      - 5+ cols: assume WMT21 (id, src, mt, toklabels, label) and use label=col4
    Returns df with columns: src, mt, label
    """
    ncols = df_raw.shape[1]
    if ncols >= 5:
        # Try 5-col WMT21 first (safe if extra columns exist)
        src  = df_raw.iloc[:, 1].astype(str)
        mt   = df_raw.iloc[:, 2].astype(str)
        lbl  = df_raw.iloc[:, 4].astype(str)
        df   = pd.DataFrame({"src": src, "mt": mt, "label": lbl})
    elif ncols >= 3:
        # 3-col generic
        src  = df_raw.iloc[:, 0].astype(str)
        mt   = df_raw.iloc[:, 1].astype(str)
        lbl  = df_raw.iloc[:, -1].astype(str)
        df   = pd.DataFrame({"src": src, "mt": mt, "label": lbl})
    else:
        raise RuntimeError(f"Expected ≥3 columns, got shape={df_raw.shape}. Check delimiter.")
    return df

def _normalize_labels_inplace(df: pd.DataFrame) -> None:
    """Upper/strip then optional BAD/OK→ERR/NOT mapping; keep only ALLOWED."""
    lab = df["label"].astype(str).str.strip().str.upper()
    mapped = lab.map(LABEL_MAP)                 # returns NaN for unknowns
    # Avoid FutureWarning by not using fillna on object w/ downcast
    df["label"] = mapped.where(mapped.notna(), lab)
    # Filter to allowed
    before = len(df)
    df.drop(index=df[~df["label"].isin(ALLOWED)].index, inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        logging.warning(f"Dropped {dropped} rows with non-{ALLOWED} labels.")
    if len(df) == 0:
        raise RuntimeError("No usable rows after label normalization (kept only ERR/NOT).")
    df["label_id"] = (df["label"] == "ERR").astype(int)

def load_data(path: str, tag: str) -> pd.DataFrame:
    """Read, normalize, and summarize a TSV split identified by ``tag``."""
    df_raw = _read_tsv(path)
    logging.info(f"[{tag}] read: rows={len(df_raw)} cols={df_raw.shape[1]}")
    df = _coerce_schema(df_raw)
    _normalize_labels_inplace(df)
    c_err = int((df["label"] == "ERR").sum())
    c_not = int((df["label"] == "NOT").sum())
    logging.info(f"[{tag}] rows={len(df)} | ERR={c_err} | NOT={c_not}")
    return df

# ── Few-shot ───────────────────────────────────────────────────────────────────
def sample_with_replace(df: pd.DataFrame, label: str, k: int, seed: int = 42) -> pd.DataFrame:
    """Return ``k`` rows for ``label`` (sampling with replacement if needed)."""
    sub = df[df["label"] == label]
    n = len(sub)
    if n == 0:
        raise RuntimeError(f"No rows for class '{label}' in TRAIN after normalization.")
    replace = n < k
    if replace:
        logging.warning(f"Class '{label}' has {n} rows; sampling {k} with replace=True.")
    return sub.sample(k, random_state=seed, replace=replace)

def select_few_shot_examples(train_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Prepare balanced few-shot exemplars from the normalized training data."""
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
    """Create the message list for a single classification request."""
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
    """Normalize raw model output into the canonical ``ERR``/``NOT`` labels."""
    if text is None:
        return "NOT"
    s = str(text).strip().upper()
    if s == "ERR": return "ERR"
    if s == "NOT": return "NOT"
    if "ERR" in s and "NOT" not in s: return "ERR"
    if "NOT" in s and "ERR" not in s: return "NOT"
    tok = s.split()[0]
    return tok if tok in ALLOWED else "NOT"

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Run few-shot inference and emit preview plus aggregate metrics."""
    dev_df   = load_data(DEV_TSV,   "DEV")
    train_df = load_data(TRAIN_TSV, "TRAIN")

    examples = select_few_shot_examples(train_df)

    gen_labels: List[str] = []
    preds: List[int] = []
    total_tokens = 0
    t0 = time.time()

    for i, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="GPT-4o few-shot inference"):
        messages = build_messages(row["src"], row["mt"], examples)
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
            logging.info(f"Row {i}: gen={lab!r} raw={raw!r} | prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
        preds.append(1 if lab == "ERR" else 0)

    elapsed = time.time() - t0
    logging.info(f"Total inference time: {elapsed:.2f}s")
    logging.info(f"Total tokens used: {total_tokens}")
    logging.info(f"Estimated cost: ${ (total_tokens/1000.0) * COST_PER_1K :.4f}")

    # Preview first 10
    print("\nFirst 10 results (Generated | True | Pred):\n")
    show_n = min(10, len(dev_df))
    for i in range(show_n):
        gen  = gen_labels[i] or "<EMPTY>"
        true = "ERR" if dev_df.iloc[i]["label_id"] == 1 else "NOT"
        pred = "ERR" if preds[i] == 1 else "NOT"
        print(f"#{i+1:2d} {gen!r} | {true} | {pred}")

    # Metrics
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
    main()
