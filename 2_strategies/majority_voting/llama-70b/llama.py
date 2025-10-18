#!/usr/bin/env python3
"""Run llama.cpp server-based few-shot CED evaluation with majority voting."""

import glob
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from tqdm import tqdm

# ===================== CONFIG =====================
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8811")
MODEL_ID_ENV = os.environ.get("MODEL_ID", "").strip()

# Point to TSV or split dirs (environment variables preferred for overrides).
TRAIN_PATH = os.environ.get("TRAIN_PATH", "/path/to/train_dataset.tsv")
DEV_PATH = os.environ.get("DEV_PATH", "/path/to/dev_dataset.tsv")

EVAL_LIMIT      = None      # None/0 for full
FEWSHOT_ERR_N   = 5
FEWSHOT_NOT_N   = 3
FEWSHOT_SEED    = 42

# Decoding (safer):
MAX_NEW_TOKENS  = 3          # <-- was 1; allow multi-token 'ERR'/'NOT'
TEMPERATURE     = 0.0
TOP_P           = 1.0
STOP_SEQ        = ["\n"]     # <-- add stop to cut cleanly

# Majority voting:
USE_MAJORITY_VOTE = True
N_VOTES           = 3
TEMP_FOR_VOTE     = 0.2
TIE_BREAK         = "NOT"
VOTE_DEBUG_PRINT  = False    # set True temporarily to inspect raw votes
assert not USE_MAJORITY_VOTE or (N_VOTES >= 1 and N_VOTES % 2 == 1), "N_VOTES must be odd >= 1"

DEFAULT_LABEL = "NOT"

SYSTEM_PROMPT = (
    "You are a precise translation evaluator.\n"
    "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
    "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
    "or only has minor imperfections.\n"
    "Do not add any explanation, punctuation, or additional text."
)

# Grammar: strictly allow only ERR or NOT
GBNF = r"""root ::= ( "ERR" | "NOT" )"""
GRAMMAR_FIELD = {"type": "gbnf", "value": GBNF}

# ---------- Loaders ----------
_LABEL_MAP = {"ERR":"ERR","NOT":"NOT","BAD":"ERR","OK":"NOT","ERROR":"ERR","CORRECT":"NOT"}

def _normalize_label(x: str) -> str:
    """Map various label spellings onto the canonical ``ERR``/``NOT`` set."""
    t = (x or "").strip().upper()
    return _LABEL_MAP.get(t, t if t in ("ERR","NOT") else "ERR")

def load_tsv_noheader(path):
    """Load a TSV without headers and coerce into ``src``, ``mt``, ``label`` columns."""
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
    df["label"] = df["label"].map(_normalize_label)
    return df[["src","mt","label"]]

def _read_lines(fp):
    """Read UTF-8 lines from ``fp`` and strip trailing newlines."""
    with open(fp, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def load_split_dir(dir_path: str) -> pd.DataFrame:
    """Load ``*.src``, ``*.mt``, and ``*.label`` files from ``dir_path``."""
    src_files   = sorted(glob.glob(os.path.join(dir_path, "*.src")))
    mt_files    = sorted(glob.glob(os.path.join(dir_path, "*.mt")))
    label_files = sorted(glob.glob(os.path.join(dir_path, "*.label")))
    if not (src_files and mt_files and label_files):
        raise FileNotFoundError(f"Missing split files in {dir_path}. Need *.src, *.mt, *.label")
    src_fp, mt_fp, label_fp = src_files[0], mt_files[0], label_files[0]
    src = _read_lines(src_fp); mt = _read_lines(mt_fp); lb = _read_lines(label_fp)
    n = min(len(src), len(mt), len(lb))
    if not (len(src) == len(mt) == len(lb)):
        print(f"[WARN] Length mismatch: src={len(src)} mt={len(mt)} label={len(lb)} → trunc {n}.", file=sys.stderr)
    return pd.DataFrame({"src": src[:n], "mt": mt[:n], "label": [_normalize_label(x) for x in lb[:n]]}, dtype=str)

def load_any_dataset(path_or_dir: str) -> pd.DataFrame:
    """Load a dataset from either a TSV file or a directory of split files."""
    p = (path_or_dir or "").strip()
    if not p: raise ValueError("Empty dataset path")
    if os.path.isdir(p): return load_split_dir(p)
    ext = os.path.splitext(p)[1].lower()
    if ext in (".tsv",".txt"): return load_tsv_noheader(p)
    if ext in (".src",".mt",".label"): return load_split_dir(os.path.dirname(p))
    return load_tsv_noheader(p)

# ---------- Few-shot & prompts ----------
def sanitize_label(text: str) -> str:
    """Normalize raw model output into an ``ERR``/``NOT`` decision."""
    if not text: return DEFAULT_LABEL
    t = (text or "").strip().upper()
    if t in ("ERR","NOT"): return t
    t2 = re.sub(r"[`\"'“”‘’]", " ", t)
    err_pos = next((m.start() for m in re.finditer(r"\bERR\b", t2)), None)
    not_pos = next((m.start() for m in re.finditer(r"\bNOT\b", t2)), None)
    if err_pos is not None and not_pos is not None:
        return "ERR" if err_pos < not_pos else "NOT"
    if err_pos is not None: return "ERR"
    if not_pos is not None: return "NOT"
    return DEFAULT_LABEL

def wait_for_server(api_base: str, timeout_s: int = 120):
    """Poll the llama.cpp server until it responds or ``timeout_s`` expires."""
    t0 = time.time(); last_err = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{api_base}/v1/models", timeout=3)
            if r.ok: return True
        except Exception as e:
            last_err = e
        time.sleep(2)
    print(f"[ERR ] Server at {api_base} not reachable after {timeout_s}s. Last error: {last_err}", file=sys.stderr)
    return False

def get_model_id(api_base: str, prefer: str | None):
    """Fetch the server model list and choose an identifier to query."""
    try:
        r = requests.get(f"{api_base}/v1/models", timeout=5); r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        if not ids: raise RuntimeError("No models returned")
        return prefer if (prefer and prefer in ids) else ids[0]
    except Exception as e:
        print(f"[WARN] /v1/models failed ({e}); using {prefer or 'local'}", file=sys.stderr)
        return prefer or "local"

def sample_fewshot(df_train: pd.DataFrame, n_err=5, n_not=3, seed=42):
    """Sample ERR/NOT exemplars with deterministic shuffling for reproducibility."""
    rnd = random.Random(seed)
    err_rows = df_train[df_train["label"] == "ERR"].copy()
    not_rows = df_train[df_train["label"] == "NOT"].copy()
    err_idx = list(err_rows.index); not_idx = list(not_rows.index)
    rnd.shuffle(err_idx); rnd.shuffle(not_idx)
    err_pick = err_rows.loc[err_idx[:min(n_err, len(err_idx))]]
    not_pick = not_rows.loc[not_idx[:min(n_not, len(not_idx))]]
    demos = []
    for r in err_pick.itertuples(index=False): demos.append((r.src, r.mt, "ERR"))
    for r in not_pick.itertuples(index=False): demos.append((r.src, r.mt, "NOT"))
    return demos

def build_messages_with_fewshot(src: str, mt: str, demos):
    """Compose chat messages that include few-shot demonstrations."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for (d_src, d_mt, d_label) in demos:
        messages.append({"role": "user",      "content": f"EN: {d_src.strip()}\nDE: {d_mt.strip()}\nLabel (ERR or NOT):"})
        messages.append({"role": "assistant", "content": d_label})
    messages.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}\nLabel (ERR or NOT):"})
    return messages

# ---------- Inference ----------
def _post_chat(payload, retries: int = 6):
    """POST to the llama.cpp server with retry/backoff for transient failures."""
    backoff = 0.5
    for attempt in range(1, retries+1):
        try:
            r = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, timeout=180)
            if r.status_code in (429,503):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (429,503) and attempt < retries:
                sleep_s = min(4.0, backoff)
                print(f"[WARN] {code} (attempt {attempt}/{retries}) — backoff {sleep_s:.1f}s")
                time.sleep(sleep_s); backoff *= 2; continue
            body = None
            try: body = e.response.text[:500]
            except Exception: pass
            raise RuntimeError(f"Server error {code}: {body}") from e
        except requests.RequestException as e:
            if attempt < retries:
                sleep_s = min(4.0, backoff)
                print(f"[WARN] Request error '{e}' (attempt {attempt}/{retries}) — retry in {sleep_s:.1f}s")
                time.sleep(sleep_s); backoff *= 2; continue
            raise

def infer_single(src, mt, model_id: str, demos):
    """Make a single constrained decode request and return the sanitized label."""
    payload = {
        "model": model_id,
        "messages": build_messages_with_fewshot(src, mt, demos),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "grammar": GRAMMAR_FIELD,
        "stop": STOP_SEQ,
    }
    data = _post_chat(payload)
    return sanitize_label(data["choices"][0]["message"]["content"])

def infer_majority(src, mt, model_id: str, demos, n_votes: int, temp_for_vote: float):
    """Run ``n_votes`` inference passes (batched when possible) and majority vote."""
    # Try efficient single-call with `n`
    payload = {
        "model": model_id,
        "messages": build_messages_with_fewshot(src, mt, demos),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": temp_for_vote,
        "top_p": TOP_P,
        "grammar": GRAMMAR_FIELD,
        "stop": STOP_SEQ,
        "n": n_votes,
    }
    try:
        data = _post_chat(payload)
        outs = [(ch.get("message", {}).get("content") or "").strip() for ch in data.get("choices", [])]
        # If server ignored `n` (gave 1 choice), fall through to sequential
        if len(outs) == n_votes:
            labels = [sanitize_label(o) for o in outs]
            tally = Counter(labels)
            if VOTE_DEBUG_PRINT:
                print(f"[VOTES] raw={outs} -> labels={labels} -> counts={dict(tally)}")
            if tally["ERR"] > tally["NOT"]: return "ERR"
            if tally["NOT"] > tally["ERR"]: return "NOT"
            return TIE_BREAK.upper()
    except Exception:
        pass  # fall back to sequential below

    # Sequential fallback (respects grammar reliably)
    labels = []
    for _ in range(n_votes):
        payload_seq = {
            "model": model_id,
            "messages": build_messages_with_fewshot(src, mt, demos),
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": temp_for_vote,
            "top_p": TOP_P,
            "grammar": GRAMMAR_FIELD,
            "stop": STOP_SEQ,
        }
        data = _post_chat(payload_seq)
        labels.append(sanitize_label(data["choices"][0]["message"]["content"]))
    tally = Counter(labels)
    if VOTE_DEBUG_PRINT:
        print(f"[VOTES*] labels={labels} -> counts={dict(tally)}")
    if tally["ERR"] > tally["NOT"]: return "ERR"
    if tally["NOT"] > tally["ERR"]: return "NOT"
    return TIE_BREAK.upper()

# -------------------- MAIN --------------------
def main():
    """Evaluate the dev set via llama.cpp and emit preview plus summary metrics."""
    if not wait_for_server(API_BASE, timeout_s=120):
        sys.exit(2)
    model_id = get_model_id(API_BASE, MODEL_ID_ENV)
    print(f"[INFO] Using model id: {model_id}")

    df_full = load_any_dataset(DEV_PATH)
    df = df_full.head(EVAL_LIMIT) if EVAL_LIMIT else df_full
    print(f"[INFO] Evaluating {len(df)} row(s) via llama-server at {API_BASE}")

    train_path = (TRAIN_PATH or "").strip() or DEV_PATH
    try:
        df_train = load_any_dataset(train_path)
    except Exception as e:
        print(f"[WARN] Failed to load TRAIN_PATH ('{train_path}'): {e}. Falling back to DEV_PATH.", file=sys.stderr)
        df_train = df_full

    demos = sample_fewshot(df_train, n_err=FEWSHOT_ERR_N, n_not=FEWSHOT_NOT_N, seed=FEWSHOT_SEED)
    err_ct = sum(1 for _,_,lab in demos if lab == "ERR")
    not_ct = sum(1 for _,_,lab in demos if lab == "NOT")
    print(f"[INFO] Few-shot prepared: {err_ct} ERR + {not_ct} NOT (order: ERR first, then NOT) from '{train_path}'")

    y_true, y_pred, latencies = [], [], []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Inference", unit="row"), 1):
        t0 = time.perf_counter()
        try:
            if USE_MAJORITY_VOTE and N_VOTES > 1:
                # Optionally inspect first few
                if VOTE_DEBUG_PRINT and i <= 5:
                    print(f"\n[ROW {i}] EN={row.src[:120]} ... | DE={row.mt[:120]} ...")
                pred = infer_majority(row.src, row.mt, model_id=model_id, demos=demos,
                                      n_votes=N_VOTES, temp_for_vote=TEMP_FOR_VOTE)
            else:
                pred = infer_single(row.src, row.mt, model_id=model_id, demos=demos)
        except Exception:
            pred = DEFAULT_LABEL
        dt  = time.perf_counter() - t0
        latencies.append(dt); y_pred.append(pred); y_true.append(row.label)
        # Light debug without flooding:
        if i <= 10:
            print(f"[DEBUG] TRUE={row.label} PRED={pred} | latency={dt:.2f}s")

    map01 = {"ERR":1, "NOT":0}
    yt = [map01.get(y,0) for y in y_true]
    yp = [map01.get(y,0) for y in y_pred]
    mcc = matthews_corrcoef(yt, yp) if len(yt) > 1 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, labels=[1,0], zero_division=0)
    acc = (pd.Series(yt) == pd.Series(yp)).mean()
    cm  = confusion_matrix(yt, yp, labels=[1,0])
    cm_df = pd.DataFrame(cm, index=["ERR_true","NOT_true"], columns=["ERR_pred","NOT_pred"])

    print("\n--- Results ---")
    print(f"Subset size: {len(df)}")
    print(f"Few-shot demos: {FEWSHOT_ERR_N} ERR + {FEWSHOT_NOT_N} NOT (rs={FEWSHOT_SEED})")
    if USE_MAJORITY_VOTE and N_VOTES > 1:
        print(f"Majority vote: {N_VOTES} samples @ T={TEMP_FOR_VOTE}")
    print(f"Accuracy: {acc:.4f}, MCC: {mcc:.4f}")
    print(f"F1-ERR: {f1[0]:.4f}, F1-NOT: {f1[1]:.4f}")
    if latencies:
        print(f"Latency (s): mean={np.mean(latencies):.2f}, max={np.max(latencies):.2f}")
    print("\nConfusion Matrix (rows=true │ cols=pred)")
    print(cm_df.to_string())

if __name__ == "__main__":
    main()
