#!/usr/bin/env python3
# GPT-OSS CED (Option A): long decode + parse FINAL label → metrics
# - Python 3.9 compatible
# - Uses your HF cache paths & SYSTEM_PROMPT
# - Few-shot optional (from TRAIN_TSV)
# - Row limit knob (PROCESS_N); set 0 for ALL rows
# - No log-prob scoring, just parsing FINAL (fallback to first ERR/NOT)
# - Majority voting (n samples per prompt) with tie-break

import os, sys, logging, re
from typing import Optional, List, Dict, Tuple
from inspect import signature
from collections import Counter

# ── Environment (same style as your Llama script) ─────────────────────────────
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("HF_HOME", "/hpcwork/ni124545/hf_cache")

import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("ced-gptoss-optionA")

# ===================== CONFIG (edit here) =====================================
MODEL_REPO      = "openai/gpt-oss-20b"  # swap to "openai/gpt-oss-8b" if VRAM is tight

CACHE_ROOT      = "/hpcwork/ni124545/hf_cache"
MODEL_LOCAL_DIR = os.path.join(CACHE_ROOT, "models", MODEL_REPO.replace("/", "_"))

DEV_TSV         = "/home/ni124545/llm/data/wmt21/ende_majority_dev.tsv"
TRAIN_TSV       = "/home/ni124545/llm/data/wmt21/ende_majority_train.tsv"

# Row limit: 0 → process ALL rows; >0 → first N rows
PROCESS_N       = 0

# Few-shot toggle and counts (like your Llama script)
USE_FEW_SHOT        = True
FEW_SHOT_ERR_CNT    = 5
FEW_SHOT_NOT_CNT    = 3
RANDOM_STATE        = 42

# Decoding: allow enough room to reach FINAL
MAX_NEW_TOKENS  = 256

# ── Majority Vote settings (mirrors your GPT‑4o script) ───────────────────────
USE_MAJORITY_VOTE = True     # set False to disable
N_VOTES           = 3        # odd number: 3 or 5 typical
TEMP_FOR_VOTE     = 0.2      # small > 0 to induce slight diversity
TIE_BREAK         = "NOT"    # "NOT" (precision-leaning) or "ERR" (recall-leaning)
# If you disable majority, we use greedy (no sampling).

# Same prompt as your Llama script
SYSTEM_PROMPT = (
    "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
    "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
    "or only has minor imperfections.\n"
    "Do not add any explanation, punctuation, or additional text."
)

# Reasoning effort hint (used if template supports it)
REASONING_EFFORT = "low"   # "low" | "medium" | "high" | None

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# ===================== Helpers ===============================================
def ensure_snapshot_local(repo_id: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    kwargs = dict(repo_id=repo_id, local_dir=local_dir, token=HF_TOKEN)
    if "use_hf_transfer" in signature(snapshot_download).parameters:
        kwargs["use_hf_transfer"] = False
    log.info(f"Snapshotting repo '{repo_id}' → {local_dir}")
    snapshot_download(**kwargs)
    log.info("Snapshot ready.")

def load_model_and_tokenizer():
    ensure_snapshot_local(MODEL_REPO, MODEL_LOCAL_DIR)

    # Try FA2; else SDPA
    use_fa2 = False
    try:
        import flash_attn  # noqa: F401
        use_fa2 = True
        log.info("flash_attn detected → using FlashAttention 2")
    except Exception:
        log.info("flash_attn not found → using PyTorch SDPA")

    tok = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_DIR, use_fast=True, trust_remote_code=True, local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if use_fa2 else "sdpa",
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
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}"},
    ]

def build_messages_fewshot(examples: List[Dict[str,str]], src: str, mt: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        msgs.append({"role": "user",      "content": f"EN: {ex['src'].strip()}\nDE: {ex['mt'].strip()}"})
        msgs.append({"role": "assistant", "content": ex['label'].strip().upper()})
    msgs.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}"})
    return msgs

def select_few_shot_examples_from_train(train_tsv: str,
                                        n_err: int, n_not: int,
                                        random_state: int = 42) -> List[Dict[str,str]]:
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
    s = (t or "").strip().upper()
    if "ERR" in s and "NOT" in s:
        return "ERR" if s.index("ERR") < s.index("NOT") else "NOT"
    if "ERR" in s: return "ERR"
    if "NOT" in s: return "NOT"
    return "ERR"  # conservative fallback

def _eos_token_ids(tok) -> Optional[List[int]]:
    ids: List[int] = []
    for tok_str in ("<|end|>", "<|return|>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        if tid is not None and tid != -1:
            ids.append(tid)
    return ids or None

@torch.inference_mode()
def generate_votes(model, tok, msgs, reasoning_effort: Optional[str]) -> List[str]:
    """
    Generate one or multiple completions and return the list of parsed labels per completion.
    - If USE_MAJORITY_VOTE and N_VOTES>1: do_sample=True, temperature=TEMP_FOR_VOTE, num_return_sequences=N_VOTES
    - Else: greedy single decode.
    """
    # Render prompt once
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

    eos_ids = _eos_token_ids(tok)

    # Decide decoding mode
    use_vote = bool(USE_MAJORITY_VOTE and N_VOTES and N_VOTES > 1)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tok.eos_token_id,
        eos_token_id=eos_ids,            # accepts int or list[int]
    )

    if use_vote:
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=float(TEMP_FOR_VOTE),
                top_p=1.0,
                num_return_sequences=int(N_VOTES),
            )
        )
    else:
        gen_kwargs.update(dict(do_sample=False))

    # Generate
    out = model.generate(**inputs, **gen_kwargs)  # shape: (num_return_sequences, ...)
    if out.dim() == 1:
        out = out.unsqueeze(0)

    # Slice off prompt for each sequence and parse
    start = inputs["input_ids"].shape[1]
    labels: List[str] = []
    for seq in out:
        gen_ids = seq[start:]
        raw = tok.decode(gen_ids, skip_special_tokens=False)
        parsed = _extract_final_or_label(raw)
        labels.append(_sanitize_label(parsed))

    return labels

def majority_decide(labels: List[str]) -> str:
    if not labels:
        return "ERR"
    if len(labels) == 1:
        return labels[0]
    c = Counter(labels)
    if c["ERR"] > c["NOT"]:
        return "ERR"
    if c["NOT"] > c["ERR"]:
        return "NOT"
    return TIE_BREAK

# ===================== Main ================================================
def main():
    log.info("Starting GPT-OSS Option-A evaluation (parse FINAL) + Majority Voting.")
    model, tok = load_model_and_tokenizer()

    df = load_tsv_noheader(DEV_TSV)
    eval_df = df if not PROCESS_N or PROCESS_N <= 0 else df.head(PROCESS_N)
    rows = list(eval_df.itertuples(index=False))
    log.info(
        f"Processing rows: {len(rows)}  |  USE_FEW_SHOT={USE_FEW_SHOT}  |  "
        f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}  |  Majority={'ON' if USE_MAJORITY_VOTE and N_VOTES>1 else 'OFF'} "
        f"(n={N_VOTES if USE_MAJORITY_VOTE else 1}, temp={TEMP_FOR_VOTE})"
    )

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

        vote_labels = generate_votes(model, tok, msgs, REASONING_EFFORT)
        pred = majority_decide(vote_labels)

        y_true.append(r.label)
        y_pred.append(pred)

        if i <= preview_k:
            vb = f" votes={vote_labels}" if len(vote_labels) > 1 else ""
            print(f"[{i:03d}] TRUE={r.label} | PRED={pred}{vb}")

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

    print(f"\nProcessed: {len(rows)} rows  |  Few-shot: {'ON' if USE_FEW_SHOT else 'OFF'}  |  Majority: {'ON' if USE_MAJORITY_VOTE and N_VOTES>1 else 'OFF'}")
    print(f"MCC   : {mcc:.4f}")
    print(f"F1-ERR: {f1_err:.4f}  F1-NOT: {f1_not:.4f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR  {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"NOT  {cm[1,0]:5d} {cm[1,1]:5d}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
