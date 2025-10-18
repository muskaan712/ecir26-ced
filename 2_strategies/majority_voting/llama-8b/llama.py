#!/usr/bin/env python3
"""Run Meta Llama 3.1 8B few-shot CED evaluation with optional majority voting."""

import os
import re
from collections import Counter

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/path/to/data")
DEV_TSV = os.environ.get("DEV_TSV", os.path.join(DATA_DIR, "ende_wmt22_dev.tsv"))
TRAIN_TSV = os.environ.get("TRAIN_TSV", os.path.join(DATA_DIR, "ende_wmt22_train.tsv"))

HF_TOKEN   = os.getenv("HF_TOKEN")
MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_ROOT = os.environ.get("CACHE_ROOT", os.path.expanduser("~/.cache"))
CACHE_DIR = os.path.join(CACHE_ROOT, MODEL_ID.replace("/", "_"))

# ─── Inference ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 2
MAX_NEW_TOKENS = 3          # ERR/NOT may tokenize to >=1 piece
MIN_NEW_TOKENS = 1
TOP_P          = 1.0

# Attn backend: keep it simple/portable
ATTN_IMPL = "eager"

# ─── Few-shot config ───────────────────────────────────────────────────────────
USE_FEW_SHOT        = True
FEW_SHOT_ERR_CNT    = 5
FEW_SHOT_NOT_CNT    = 3
RANDOM_STATE        = 42

# ─── Majority voting ───────────────────────────────────────────────────────────
USE_MAJORITY_VOTE = True   # set False for deterministic single sample
N_VOTES           = 3      # odd number (3 or 5 typical)
TEMP_FOR_VOTE     = 0.2    # mild diversity
TIE_BREAK         = "NOT"  # specificity-leaning
assert not USE_MAJORITY_VOTE or (N_VOTES >= 1 and N_VOTES % 2 == 1), "N_VOTES must be odd >= 1"

# When not using majority voting:
DO_SAMPLE_SINGLE   = False  # deterministic greedy for single-sample path

# ─── Prompt text ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a STRICT binary classifier for WMT’21 Task 3 (Critical Error Detection, EN→DE).\n"
    "Output EXACTLY one token: ERR or NOT (UPPERCASE, no punctuation, no explanation).\n\n"
    "ERR if ANY critical meaning deviation exists:\n"
    "• TOX: toxicity/hate/violence/profanity introduced, deleted, mistranslated, or left untranslated.\n"
    "• SAF: health/safety risk introduced, deleted, mistranslated, or left untranslated.\n"
    "• NAM: named entity added/removed/mistranslated/gibberish/unrecoverable transliteration.\n"
    "• SEN: polarity/negation flipped or materially strengthened/weakened.\n"
    "• NUM: wrong/missing/added numbers, dates, times, units that change meaning.\n\n"
    "NOT for non-critical issues: style/register/awkwardness, locale punctuation/casing/diacritics,\n"
    "minor wording that keeps meaning, and correct transfer of toxicity from source.\n\n"
    "Decision policy: If uncertain, choose NOT.\n"
    "Return ONLY: ERR or NOT."
)

# ───────────────────────────────────────────────────────────────────────────────
def download_and_cache_model():
    """Download MODEL_ID into CACHE_DIR once, then reuse."""
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=CACHE_DIR,
            token=HF_TOKEN,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.md", "*.py"]
        )

def _map_label_to_err_not(x: str) -> str:
    """Normalize labels to {'ERR','NOT'}; handle BAD/OK and noisy variants."""
    s = str(x or "").strip().upper()
    if s == "BAD": return "ERR"
    if s == "OK":  return "NOT"
    if s in ("ERR", "NOT"): return s
    # If both appear, honor first occurrence
    err_i = s.find("ERR")
    not_i = s.find("NOT")
    if err_i != -1 and not_i != -1:
        return "ERR" if err_i < not_i else "NOT"
    if err_i != -1: return "ERR"
    if not_i != -1: return "NOT"
    return "ERR"  # conservative

def _split_tsv_flex(path: str):
    """
    Read TSV and return (src, mt, label) as strings.
    Supports:
      3 cols: src, mt, label
      4 cols: src, mt, _, label
      ≥5 cols: id, src, mt, _, label (WMT21-style)
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, na_filter=False)
    n = df.shape[1]
    if n < 3:
        raise ValueError(f"Expected ≥3 columns (src, mt, label). Got {n} in {path}")
    if n == 3:
        src = df.iloc[:, 0]; mt = df.iloc[:, 1]; label = df.iloc[:, 2]
    elif n == 4:
        src = df.iloc[:, 0]; mt = df.iloc[:, 1]; label = df.iloc[:, 3]
    else:  # n >= 5
        src = df.iloc[:, 1]; mt = df.iloc[:, 2]; label = df.iloc[:, 4]
    src   = src.astype(str)
    mt    = mt.astype(str)
    label = label.astype(str).map(_map_label_to_err_not)
    return src, mt, label

def load_dev(path: str) -> pd.DataFrame:
    """Load the evaluation split and attach numeric labels."""
    src, mt, label = _split_tsv_flex(path)
    out = pd.DataFrame({"src": src, "mt": mt, "label": label})
    out["label_id"] = out["label"].map({"ERR": 1, "NOT": 0})
    return out

def load_train_minimal(path: str) -> pd.DataFrame:
    """Load the training data into a minimal dataframe for sampling."""
    src, mt, label = _split_tsv_flex(path)
    return pd.DataFrame({"src": src, "mt": mt, "label": label})

def sample_few_shot_examples(train_tsv: str,
                             n_err: int,
                             n_not: int,
                             random_state: int = 42):
    """Return a list of few-shot exemplars sampled from the training TSV."""
    train_df = load_train_minimal(train_tsv)
    train_df = train_df[train_df["label"].isin(["ERR", "NOT"])]

    def _sample(df, k):
        if len(df) == 0: return df
        if len(df) >= k: return df.sample(k, random_state=random_state)
        return df.sample(k, replace=True, random_state=random_state)

    err_df = _sample(train_df[train_df["label"] == "ERR"], n_err)
    not_df = _sample(train_df[train_df["label"] == "NOT"], n_not)

    examples = []
    for _, r in err_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "ERR"})
    for _, r in not_df.iterrows():
        examples.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "NOT"})
    return examples

def build_messages_zero_shot(src: str, mt: str):
    """Create a zero-shot chat prompt for a single classification."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}\nAnswer with ONLY 'ERR' or 'NOT'."}
    ]

def build_messages_few_shot(examples, src: str, mt: str):
    """Construct chat messages that include each few-shot demonstration."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user",      "content": f"EN: {ex['src']}\nDE: {ex['mt']}"})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}\nAnswer with ONLY 'ERR' or 'NOT'."})
    return msgs

def extract_label(text: str) -> str:
    """Extract final 'ERR' or 'NOT'. Default to NOT if ambiguous/empty."""
    t = (text or "").upper()
    hits = re.findall(r"\b(ERR|NOT)\b", t)
    return hits[-1] if hits else "NOT"

def main():
    """Download the model (if needed), run inference, and print summary metrics."""
    # 1) Cache model locally
    download_and_cache_model()

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(CACHE_DIR, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # better for decoder-only batch gen

    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        CACHE_DIR,
        local_files_only=True,
        device_map="auto",
        torch_dtype=dtype
    ).eval()

    # Force eager attention impl if supported
    try:
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = ATTN_IMPL
    except Exception:
        pass

    # 3) Load data
    df = load_dev(DEV_TSV)

    # 4) Few-shot exemplars (no leakage into eval)
    few_shots = None
    if USE_FEW_SHOT:
        few_shots = sample_few_shot_examples(TRAIN_TSV, FEW_SHOT_ERR_CNT, FEW_SHOT_NOT_CNT, RANDOM_STATE)

    # 5) Generation
    gen_texts = []
    preds = []

    # stop on EOS and Llama <|eot_id|>
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            eos_ids.append(eot_id)
    except Exception:
        pass

    total = len(df)
    bar_desc = f"{'Few-shot' if USE_FEW_SHOT else 'Zero-shot'} (Llama 3.1-8B)"
    for start in tqdm(range(0, total, BATCH_SIZE), desc=bar_desc):
        end = min(start + BATCH_SIZE, total)
        rows = df.iloc[start:end]

        # Build chat prompts (strings) using official template
        prompts = []
        for _, r in rows.iterrows():
            msgs = (build_messages_few_shot(few_shots, r.src, r.mt)
                    if USE_FEW_SHOT and few_shots else
                    build_messages_zero_shot(r.src, r.mt))
            s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompts.append(s)

        # Batch tokenization
        enc = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids     = enc["input_ids"].to(model.device)
        attention_mask= enc["attention_mask"].to(model.device)

        # Per-row prompt lengths (non-pad tokens)
        prompt_lens = attention_mask.sum(dim=1).tolist()
        pad_id = tokenizer.pad_token_id

        with torch.no_grad():
            if USE_MAJORITY_VOTE and N_VOTES > 1:
                # Multi-sample: outputs shape = (B * N_VOTES, seq_len)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MIN_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMP_FOR_VOTE,
                    top_p=TOP_P,
                    num_return_sequences=N_VOTES,
                    eos_token_id=eos_ids,
                    pad_token_id=pad_id
                )
                B = rows.shape[0]
                seq_len = outputs.size(1)

                # Regroup by row: for each row i, slice N_VOTES samples
                for i in range(B):
                    start_i = i * N_VOTES
                    end_i   = start_i + N_VOTES
                    p_len   = int(prompt_lens[i])
                    vote_labels, vote_texts = [], []
                    for j in range(start_i, end_i):
                        # Slice only the generated continuation for THIS row
                        gen_ids = outputs[j, p_len:seq_len]
                        txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                        if not txt:
                            txt = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
                        lab = extract_label(txt)
                        vote_texts.append(txt)
                        vote_labels.append(lab)
                    tally = Counter(vote_labels)
                    if tally["ERR"] > tally["NOT"]:
                        chosen = "ERR"
                    elif tally["NOT"] > tally["ERR"]:
                        chosen = "NOT"
                    else:
                        chosen = TIE_BREAK.upper()
                    gen_texts.append(chosen)
                    preds.append(1 if chosen == "ERR" else 0)
            else:
                # Single deterministic path
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MIN_NEW_TOKENS,
                    do_sample=DO_SAMPLE_SINGLE,
                    temperature=0.0 if not DO_SAMPLE_SINGLE else None,
                    top_p=TOP_P if DO_SAMPLE_SINGLE else None,
                    eos_token_id=eos_ids,
                    pad_token_id=pad_id
                )
                seq_len = outputs.size(1)
                for i in range(rows.shape(0) if callable(rows.shape) else rows.shape[0]):
                    p_len = int(prompt_lens[i])
                    gen_ids = outputs[i, p_len:seq_len]
                    txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    if not txt:
                        txt = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
                    gen_texts.append(txt)
                    lbl = extract_label(txt)
                    preds.append(1 if lbl == "ERR" else 0)

    # 6) First 10 preview
    print("\nFirst 10 (generated → true / pred):\n")
    for i in range(min(10, len(df))):
        gt = gen_texts[i] if gen_texts[i] else "<EMPTY>"
        true = "ERR" if df.loc[i, 'label_id'] == 1 else "NOT"
        pred = "ERR" if preds[i] == 1 else "NOT"
        print(f"#{i+1:2d} Generated: {gt!r}")
        print(f"    True / Pred: {true} / {pred}\n")

    # 7) Metrics
    labels = df["label_id"].tolist()
    mcc    = matthews_corrcoef(labels, preds)
    prf    = precision_recall_fscore_support(labels, preds, labels=[1,0], zero_division=0)
    cm     = confusion_matrix(labels, preds, labels=[1,0])

    print(f"\nMCC   : {mcc:.4f}")
    print(f"F1-ERR: {prf[2][0]:.4f}")
    print(f"F1-NOT: {prf[2][1]:.4f}\n")
    print("Confusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"NOT {cm[1,0]:6d} {cm[1,1]:6d}")

if __name__ == "__main__":
    main()
