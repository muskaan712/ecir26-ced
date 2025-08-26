#!/usr/bin/env python3
# infer_ft_4omini_vote_fallback.py
# Use your fine-tuned 4o-mini model with n=3 majority vote + 2–1 fallback → 4o.

import os
import pandas as pd
from tqdm import tqdm
import time
import logging
import openai
from collections import Counter
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
logging.basicConfig(
    filename="inference_ft_4omini.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = os.path.join(DATA_DIR, "ende_majority_dev.tsv")

# Ensure your OpenAI API key is set in the environment (export OPENAI_API_KEY or OAPI).
openai.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OAPI")

# Models
MODEL_MAIN       = "ft:gpt-4o-mini-2024-07-18:fraunhofer-iais:ced-label-only-50pct-v1:C33F0sry"  # <-- your FT model
MODEL_FALLBACK   = "gpt-4o"        # used only on 2–1 split votes

# Generation settings
MAX_TOKENS       = 1               # label is a single token ('ERR' or 'NOT')
TEMPERATURE      = 0.0
STOP_SEQ         = ["\n"]          # Extra safety

# Cost estimate per 1K tokens (placeholder; adjust if you want accurate billing est.)
COST_PER_1K      = 0.0025

# Keep inference aligned with FT: no few-shot, short instruction
USE_FEW_SHOT     = False

# Majority vote settings
USE_MAJORITY_VOTE = True
N_VOTES            = 3             # single request with 3 choices
TEMP_FOR_VOTE      = 0.2           # slight diversity for voting
TIE_BREAK          = "NOT"         # "NOT" for precision-leaning; "ERR" for recall-leaning

# Adaptive fallback
ADAPTIVE_FALLBACK       = False     # enable second call only on 2–1 splits
HARD_CAP_FALLBACK_PCT   = 0.0     # never escalate >25% of rows; set to 0.00 to force zero fallbacks

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


# Short prompt = same as FT setup
FT_SYSTEM_PROMPT = (
    "You are a STRICT binary classifier for EN→DE critical error detection.\n"
    "Decide if the MT contains a CRITICAL meaning error vs the source.\n"
    "Output EXACTLY one token: ERR or NOT."
)

# (kept here in case you want to switch back)
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
    system_prompt = FT_SYSTEM_PROMPT
    if USE_FEW_SHOT:
        system_prompt += "\n\n" + FEW_SHOT_BLOCK

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
    if "ERR" in t and "NOT" not in t:
        return "ERR"
    if "NOT" in t and "ERR" not in t:
        return "NOT"
    return "NOT"


def main():
    """Run inference over the dataset and report metrics."""
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not found. Set env var OPENAI_API_KEY or OAPI.")

    df = load_data(DEV_TSV)
    total_rows = len(df)

    gen_labels = []
    preds = []

    # Track usage separately
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Safety / accounting
    fallback_calls = 0
    api_calls = 0

    start_time = time.time()

    for i, row in tqdm(df.iterrows(), total=total_rows, desc="FT 4o-mini inference (vote + fallback)"):
        messages = build_messages(row.src, row.mt)

        if USE_MAJORITY_VOTE and N_VOTES > 1:
            # One request with multiple choices
            resp = openai.chat.completions.create(
                model=MODEL_MAIN,
                messages=messages,
                temperature=TEMP_FOR_VOTE,
                max_tokens=MAX_TOKENS,
                n=N_VOTES,            # <-- single request returns 3 choices
                stop=STOP_SEQ
            )
            api_calls += 1

            raws = [(c.message.content or "").strip() for c in resp.choices]
            labels = [sanitize_label(r) for r in raws]
            counts = Counter(labels)

            # majority decision
            if counts["ERR"] > counts["NOT"]:
                text = "ERR"
            elif counts["NOT"] > counts["ERR"]:
                text = "NOT"
            else:
                text = TIE_BREAK  # rare exact tie

            usage = resp.usage
            logging.info(
                f"Row {i}: vote_raws={raws!r}, vote_labels={labels!r}, "
                f"tally={dict(counts)}, chosen={text}"
            )

            # Adaptive fallback ONLY on 2–1 splits, with hard cap
            if ADAPTIVE_FALLBACK and abs(counts["ERR"] - counts["NOT"]) == 1:
                if (fallback_calls + 1) / total_rows <= HARD_CAP_FALLBACK_PCT:
                    resp2 = openai.chat.completions.create(
                        model=MODEL_FALLBACK,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=MAX_TOKENS,
                        n=1,
                        stop=STOP_SEQ
                    )
                    api_calls += 1
                    fallback_calls += 1
                    raw2 = (resp2.choices[0].message.content or "").strip()
                    text = sanitize_label(raw2)

                    # Add fallback usage to totals
                    if getattr(resp2, "usage", None):
                        total_prompt_tokens     += getattr(resp2.usage, "prompt_tokens", 0) or 0
                        total_completion_tokens += getattr(resp2.usage, "completion_tokens", 0) or 0
                else:
                    logging.info(
                        f"Row {i}: fallback skipped (cap reached). "
                        f"fallback_calls={fallback_calls}, cap={HARD_CAP_FALLBACK_PCT}"
                    )
        else:
            # Single label inference
            resp = openai.chat.completions.create(
                model=MODEL_MAIN,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n=1,
                stop=STOP_SEQ
            )
            api_calls += 1

            raw = (resp.choices[0].message.content or "").strip()
            text = sanitize_label(raw)
            usage = resp.usage
            logging.info(f"Row {i}: raw={raw!r}, used={text}")

        gen_labels.append(text)

        # Accumulate token counts for the main request (usage aggregated across n choices)
        if usage:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            total_prompt_tokens += pt
            total_completion_tokens += ct

        pred = 1 if text == "ERR" else 0
        preds.append(pred)

        # Optional: per-row token logging
        logging.info(
            f"Row {i}: prompt_tokens={getattr(usage, 'prompt_tokens', None)}, "
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

    # Safety summary
    print(f"\nFallback escalations: {fallback_calls}/{total_rows} "
          f"({100.0 * fallback_calls / total_rows:.1f}%)")
    print(f"Total API requests:   {api_calls}")

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
