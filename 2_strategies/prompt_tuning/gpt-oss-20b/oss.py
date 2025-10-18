#!/usr/bin/env python3
"""Run GPT-OSS 20B prompt-tuning evaluation with FINAL-channel parsing."""

import logging
import os
import re
import sys
from inspect import signature
from typing import Dict, List, Optional

# ── Minimal, clean env ─────────────────────────────────────────────────────────
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/path/to/hf_cache"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_QUANTIZATION_METHOD", "none")  # don't try MXFP4 etc.

import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

# Speed niceties (safe on H100)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("ced-gptoss-optionA-zeroshot")

# ===================== CONFIG (edit here) =====================================
MODEL_REPO = "openai/gpt-oss-20b"  # swap to "openai/gpt-oss-8b" if VRAM is tight
CACHE_ROOT = os.environ.get("CACHE_ROOT", os.path.expanduser("~/.cache"))
MODEL_LOCAL_DIR = os.path.join(CACHE_ROOT, "models", MODEL_REPO.replace("/", "_"))

DEV_TSV = os.environ.get("DEV_TSV", "/path/to/dev_dataset.tsv")
TRAIN_TSV = os.environ.get("TRAIN_TSV", "/path/to/train_dataset.tsv")  # unused in zero-shot

# Row limit: 0 → process ALL rows; >0 → first N rows
PROCESS_N       = 0

# Zero-shot toggle (keep False)
USE_FEW_SHOT        = True
FEW_SHOT_ERR_CNT    = 5
FEW_SHOT_NOT_CNT    = 3
RANDOM_STATE        = 42

# Decoding: allow enough room to reach FINAL; greedy (no sampling)
MAX_NEW_TOKENS  = 256

SYSTEM_PROMPT = (
   """
You are a bilingual (EN→DE) translation safety checker for sentence-level Critical Error Detection (CED).

Task
- Compare EN (source) with DE (translation).
- Decide if DE deviates in meaning from EN in a way that could mislead with real-world consequences.
- Use private reasoning, then output a single uppercase label.

How to think (channels)
- Use the <analysis> channel to reason briefly.
- In the <final> channel output EXACTLY one token: ERR or NOT (no punctuation, no extra text).

Mark ERR if at least one category below is triggered by translation error (introduced, deleted, mistranslated, or gibberish):
• TOX — Toxicity (hate/violence/profanity) meaning deviates. If toxicity in EN is preserved in DE, this is NOT an error.
• SAF — Health/safety risk meaning deviates (e.g., warnings removed, unsafe action introduced).
• NAM — Named entity deviation: person/org/place/username added/removed/changed, left untranslated, or nonsensical transliteration. Minor variations that still clearly refer to the same entity → NOT.
• SEN — Sentiment/negation flip (negation added/removed; polarity reversed; soft modal strengthened/weakened into different meaning, e.g., “possibly”→“with certainty”).
• NUM — Numbers/units/time/date changed or dropped. Unit conversions are OK only if the numeric value is correctly converted (e.g., 0 °C ↔ 32 °F).

Do NOT mark ERR for non-critical issues when meaning is preserved:
style/awkwardness; locale conventions; minor grammar/typos/casing/punctuation; small paraphrases (e.g., “my page”→“the page”) that do not change meaning.

Decision rule
- Choose ERR only if you can identify a specific word/phrase/number/entity in DE that causes a clear deviation in one of {TOX, SAF, NAM, SEN, NUM}.
- Otherwise choose NOT. When uncertain, prefer NOT.

Answer format
<analysis>
Checklist: TOX? SAF? NAM? SEN? NUM? Note any trigger terms/entities/numbers if present.
</analysis>
<final>
ERR
</final>
"""
)

# Reasoning effort hint (used if the tokenizer's chat template supports it)
REASONING_EFFORT = "low"   # "low" | "medium" | "high" | None

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# ===================== Helpers ===============================================
def ensure_snapshot_local(repo_id: str, local_dir: str):
    """Download ``repo_id`` into ``local_dir`` when it is not already cached."""
    os.makedirs(local_dir, exist_ok=True)
    kwargs = dict(repo_id=repo_id, local_dir=local_dir, token=HF_TOKEN)
    if "use_hf_transfer" in signature(snapshot_download).parameters:
        kwargs["use_hf_transfer"] = False
    log.info(f"Snapshotting repo '{repo_id}' → {local_dir}")
    snapshot_download(**kwargs)
    log.info("Snapshot ready.")

def load_model_and_tokenizer():
    """Load the GPT-OSS model and tokenizer from the local snapshot."""
    ensure_snapshot_local(MODEL_REPO, MODEL_LOCAL_DIR)

    # Try FlashAttention2; else force eager (NO SDPA — GPT-OSS not supported on SDPA yet)
    use_fa2 = False
    try:
        import flash_attn  # noqa: F401
        use_fa2 = True
        log.info("flash_attn detected → using FlashAttention 2")
    except Exception:
        log.info("flash_attn not found → using eager attention")

    tok = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_DIR, use_fast=True, trust_remote_code=True, local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=("flash_attention_2" if use_fa2 else "eager"),
        trust_remote_code=True,
        local_files_only=True,
    )

    mdl.config.use_cache = True
    try:
        mdl = torch.compile(mdl, mode="reduce-overhead", fullgraph=False)
        log.info("torch.compile applied.")
    except Exception:
        pass
    return mdl, tok

def load_tsv_noheader(path: str) -> pd.DataFrame:
    """Load a TSV file without headers and normalize to ``src``, ``mt``, ``label``."""
    log.info(f"Loading TSV: {path}")
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

def build_messages_zero_shot(src: str, mt: str) -> List[Dict[str, str]]:
    """Create a zero-shot chat prompt for a single sentence pair."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}"},
    ]

def build_messages_fewshot(
    examples: List[Dict[str, str]],
    src: str,
    mt: str,
) -> List[Dict[str, str]]:
    """Compose chat messages that include the provided few-shot exemplars."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user",      "content": f"EN: {ex['src'].strip()}\nDE: {ex['mt'].strip()}"})
        msgs.append({"role": "assistant", "content": ex['label'].strip().upper()})
    msgs.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}"})
    return msgs

def select_few_shot_examples_from_train(
    train_tsv: str,
    n_err: int,
    n_not: int,
    random_state: int = 42,
) -> List[Dict[str, str]]:
    """Sample ERR/NOT exemplars from the training TSV for prompting."""
    df = load_tsv_noheader(train_tsv)
    df = df[df["label"].isin(["ERR","NOT"])]

    def _sample(d, k):
        if len(d) == 0: return d
        if len(d) >= k: return d.sample(k, random_state=random_state)
        return d.sample(k, replace=True, random_state=random_state)

    err_df = _sample(df[df["label"] == "ERR"], n_err)
    not_df = _sample(df[df["label"] == "NOT"], n_not)

    ex = []
    for _, r in err_df.iterrows():
        ex.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "ERR"})
    for _, r in not_df.iterrows():
        ex.append({"src": r["src"].strip(), "mt": r["mt"].strip(), "label": "NOT"})
    log.info(f"Few-shot demos prepared: {len(err_df)} ERR + {len(not_df)} NOT")
    return ex

def _extract_final_or_label(decoded: str) -> str:
    """
    Extract the label from the FINAL channel if present; else first ERR/NOT in the stream.
    Keep skip_special_tokens=False when decoding so tags are visible.
    """
    m = re.search(r"<\|channel\|\>final<\|message\|\>(.*?)(?:<\|end\|\>|<\|return\|\>|$)", decoded, flags=re.S)
    if m:
        final_text = m.group(1).strip()
        m2 = re.search(r"\b(ERR|NOT)\b", final_text)
        return m2.group(1) if m2 else final_text
    m3 = re.search(r"\b(ERR|NOT)\b", decoded)
    return m3.group(1) if m3 else decoded.strip()

def _sanitize_label(t: str) -> str:
    """Robustly coerce any text to 'ERR' or 'NOT' with conservative fallback."""
    s = str(t or "").strip().upper()
    err_pos = s.find("ERR")
    not_pos = s.find("NOT")
    if err_pos != -1 and not_pos != -1:
        return "ERR" if err_pos < not_pos else "NOT"
    if err_pos != -1:
        return "ERR"
    if not_pos != -1:
        return "NOT"
    return "ERR"  # conservative fallback

@torch.inference_mode()
def generate_label(model, tok, msgs, reasoning_effort: Optional[str] = None) -> str:
    """Generate a label using the model and sanitize the resulting text."""
    # Render prompt (pass reasoning_effort if template supports it)
    try:
        prompt_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            reasoning_effort=reasoning_effort
        )
    except TypeError:
        prompt_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    enc = tok(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    # Optional: stop early when the model closes the final message
    eos_list: List[int] = []
    for tok_str in ("<|end|>", "<|return|>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        if tid is not None and tid != -1:
            eos_list.append(tid)
    eos_arg = eos_list[0] if len(eos_list) == 1 else (eos_list if eos_list else None)

    # Greedy long decode; we want to see FINAL block
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    if eos_arg is not None:
        gen_kwargs["eos_token_id"] = eos_arg

    out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    raw = tok.decode(gen_ids, skip_special_tokens=False)  # keep channel tags
    parsed = _extract_final_or_label(raw)
    return _sanitize_label(parsed)

def main():
    """Evaluate GPT-OSS predictions and report preview plus summary metrics."""
    log.info("Starting GPT-OSS Option-A evaluation (parse FINAL) — ZERO-SHOT.")
    model, tok = load_model_and_tokenizer()

    df = load_tsv_noheader(DEV_TSV)
    eval_df = df if not PROCESS_N or PROCESS_N <= 0 else df.head(PROCESS_N)
    rows = list(eval_df.itertuples(index=False))
    log.info(f"Processing rows: {len(rows)}  |  USE_FEW_SHOT={USE_FEW_SHOT}  |  MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

    few_shots = None
    if USE_FEW_SHOT:
        few_shots = select_few_shot_examples_from_train(
            TRAIN_TSV, FEW_SHOT_ERR_CNT, FEW_SHOT_NOT_CNT, RANDOM_STATE
        )

    y_true, y_pred = [], []
    preview_k = min(10, len(rows))
    print(f"\n=== PREVIEW (first {preview_k}) ===")

    for i, r in enumerate(tqdm(rows, desc="Evaluating", unit="row"), 1):
        msgs = build_messages_fewshot(few_shots, r.src, r.mt) if USE_FEW_SHOT else build_messages_zero_shot(r.src, r.mt)
        pred = generate_label(model, tok, msgs, REASONING_EFFORT)
        y_true.append(r.label)
        y_pred.append(pred)
        if i <= preview_k:
            print(f"[{i:03d}] TRUE={r.label} | PRED={pred}")

        # Optional heartbeat every 200 rows
        if i % 200 == 0:
            acc_partial = (pd.Series([1 if t=='ERR' else 0 for t in y_true]) ==
                           pd.Series([1 if p=='ERR' else 0 for p in y_pred])).mean()
            log.info(f"Progress: {i}/{len(rows)}  partial_acc={acc_partial:.3f}")

    # Metrics
    yt = [1 if t == "ERR" else 0 for t in y_true]
    yp = [1 if p == "ERR" else 0 for p in y_pred]

    mcc = matthews_corrcoef(yt, yp) if len(yt) > 1 else 0.0
    f1_err = f1_score(yt, yp, pos_label=1, zero_division=0)
    f1_not = f1_score(yt, yp, pos_label=0, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[1,0])

    print(f"\nProcessed: {len(rows)} rows  |  Few-shot: {'ON' if USE_FEW_SHOT else 'OFF'}")
    print(f"MCC   : {mcc:.4f}")
    print(f"F1-ERR: {f1_err:.4f}  F1-NOT: {f1_not:.4f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR  {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"NOT  {cm[1,0]:5d} {cm[1,1]:5d}")

if __name__ == "__main__":
    main()
