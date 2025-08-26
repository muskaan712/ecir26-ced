#!/usr/bin/env python3
# zero_shot_gpt4o_inference.py

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

# ──────────────────────────────────────────────────────────────────────────────
# Logging
logging.basicConfig(
    filename="inference_gpt4o_zero_shot.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = os.path.join(DATA_DIR, "ende_majority_dev.tsv")

# Ensure your OpenAI API key is set in the environment (export OAPI=...).
openai.api_key = os.getenv("OAPI")

MODEL          = "gpt-4o"   # You are using GPT-4o
MAX_TOKENS     = 3          # Only need 1–3 tokens for 'ERR' or 'NOT'
TEMPERATURE    = 0.0
STOP_SEQ       = ["\n"]     # Extra safety
# Adjust to your actual pricing if you want accurate estimates.
COST_PER_1K    = 0.0025

# Optional: enable an ultra-compact few-shot calibration to anchor behavior.
USE_FEW_SHOT   = True
# ───────────────────────────────────────────────────────────────────────────────


def load_data(path):
    """Load TSV data into a DataFrame and map labels to numeric IDs."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["id", "src", "mt", "toklabels", "label"],
        dtype={"id": str}
    )
    df["label_id"] = df["label"].map({"ERR": 1, "NOT": 0})
    return df


FEW_SHOT_BLOCK = (
    "Examples\n"
    "Q:\n"
    "Source (EN): Don't buy this product!\n"
    "MT (DE): Kauf dieses Produkt!\n"
    "A: ERR\n\n"
    "Q:\n"
    "Source (EN): LG releases a new smartphone.\n"
    "MT (DE): LG veröffentlicht ein neues Smartphone.\n"
    "A: NOT\n\n"
    "Q:\n"
    "Source (EN): The meeting is at 8am.\n"
    "MT (DE): Das Treffen ist um 20 Uhr.\n"
    "A: ERR\n\n"
    "Q:\n"
    "Source (EN): Hör auf, meine Seite zu bearbeiten! Lass sie verdammt nochmal in Ruhe!\n"
    "MT (DE): Hör auf, meine Seite zu bearbeiten! Lass sie in Ruhe!\n"
    "A: ERR\n\n"
    "Q:\n"
    "Source (EN): We thanked him heartily.\n"
    "MT (DE): Wir dankten ihm von Herzen.\n"
    "A: NOT\n\n"
)


def build_messages(src: str, mt: str):
    """Construct chat messages for classification tasks."""
    # WMT’21 Task 3 — STRICT binary classifier prompt (ERR/NOT)
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
    if USE_FEW_SHOT:
        system_prompt += FEW_SHOT_BLOCK

    user_prompt = (
        f"Source (EN): {src.strip()}\n"
        f"MT (DE): {mt.strip()}\n\n"
        "Label (ERR or NOT):"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def sanitize_label(text: str) -> str:
    """
    Enforce label-only output. Accept only 'ERR' or 'NOT'.
    Try to salvage if model adds extra text.
    """
    if not text:
        return "NOT"  # conservative default for MCC
    t = text.strip().upper()
    if t in ("ERR", "NOT"):
        return t
    # Salvage common variants
    if "ERR" in t and "NOT" not in t:
        return "ERR"
    if "NOT" in t and "ERR" not in t:
        return "NOT"
    # Last resort: default NOT (per decision policy)
    return "NOT"


def main():
    """Run inference over the dataset and report metrics."""
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not found. Set env var OAPI.")

    df = load_data(DEV_TSV)
    gen_labels = []
    preds = []

    # NEW: track usage separately
    total_prompt_tokens = 0
    total_completion_tokens = 0

    start_time = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="GPT-4o zero-shot inference"):
        messages = build_messages(row.src, row.mt)
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=1,
            stop=STOP_SEQ
        )
        raw = (resp.choices[0].message.content or "").strip()
        text = sanitize_label(raw)
        gen_labels.append(text)

        # UPDATED: accumulate token counts
        usage = resp.usage
        if usage:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            total_prompt_tokens += pt
            total_completion_tokens += ct

        pred = 1 if text == "ERR" else 0
        preds.append(pred)
        logging.info(
            f"Row {i}: raw={raw!r}, used={text}, "
            f"prompt_tokens={getattr(usage, 'prompt_tokens', None)}, "
            f"completion_tokens={getattr(usage, 'completion_tokens', None)}"
        )

    end_time = time.time()

    # Print + log usage summary
    total_tokens = total_prompt_tokens + total_completion_tokens
    cost = (total_tokens / 1000.0) * COST_PER_1K
    print("\nToken usage summary:")
    print(f"  Prompt tokens     : {total_prompt_tokens}")
    print(f"  Completion tokens : {total_completion_tokens}")
    print(f"  Total tokens      : {total_tokens}")
    print(f"Estimated cost      : ${cost:.4f}")
    logging.info(f"Total inference time: {end_time - start_time:.2f}s")
    logging.info(f"Prompt tokens: {total_prompt_tokens}, "
                 f"Completion tokens: {total_completion_tokens}, "
                 f"Total: {total_tokens}, Estimated cost: ${cost:.4f}")


    # Preview first 10 outputs
    print("\nFirst 10 results:\n")
    for i in range(min(10, len(gen_labels))):
        gen = gen_labels[i] or "<EMPTY>"
        true = 'ERR' if df.loc[i, 'label_id'] == 1 else 'NOT'
        pred = 'ERR' if preds[i] == 1 else 'NOT'
        print(f"#{i+1:2d} Generated: {gen!r} | True: {true} | Pred: {pred}")

    # Compute metrics
    labels = df['label_id'].tolist()
    mcc = matthews_corrcoef(labels, preds)
    prf = precision_recall_fscore_support(labels, preds, labels=[1, 0], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[1, 0])

    print(f"\nMCC   : {mcc:.4f}")
    print(f"F1-ERR: {prf[2][0]:.4f}")
    print(f"F1-NOT: {prf[2][1]:.4f}\n")
    print("Confusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"NOT {cm[1,0]:6d} {cm[1,1]:6d}")


if __name__ == "__main__":
    main()
