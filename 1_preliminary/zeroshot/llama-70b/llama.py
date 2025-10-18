#!/usr/bin/env python3
"""Zero-shot CED evaluation using a llama.cpp server backend."""

import os
import sys
import time

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

# ===================== CONFIG =====================
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8811")
# MODEL_ID will be auto-filled from /v1/models if not set or invalid
MODEL_ID_ENV = os.environ.get("MODEL_ID", "").strip()
DEV_TSV  = os.environ.get("DEV_TSV", "/path/to/dev_dataset.tsv")

EVAL_LIMIT = None  # set to None or 0 for full dataset

MAX_NEW_TOKENS = 1
TEMPERATURE    = 0.0
TOP_P          = 1.0

SYSTEM_PROMPT = (
        "You are a precise translation evaluator.\n"
        "Given an English sentence (EN) and its German translation (DE), respond with exactly one token: "
        "'ERR' if DE has a major error (meaning shift, omission, or inaccuracy), or 'NOT' if it is accurate "
        "or only has minor imperfections.\n"
        "Do not add any explanation, punctuation, or additional text."
)

# Grammar: strictly allow only ERR or NOT
GBNF = r"""root ::= ( "ERR" | "NOT" )"""
GRAMMAR_FIELD = {"type": "gbnf", "value": GBNF}  # more compatible across server builds
# ==================================================

def build_messages(src: str, mt: str):
    """Create a llama.cpp-compatible chat payload for a single example."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"EN: {src.strip()}\nDE: {mt.strip()}"},
    ]

def load_tsv_noheader(path):
    """Load TSV rows without headers and expose src/mt/label columns."""
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

def sanitize_label(text: str) -> str:
    """Normalise any model response to the ERR/NOT label set."""
    t = text.strip().upper()
    if "ERR" in t and "NOT" in t:
        return "ERR" if t.index("ERR") < t.index("NOT") else "NOT"
    if "ERR" in t: return "ERR"
    if "NOT" in t: return "NOT"
    return "ERR"

def wait_for_server(api_base: str, timeout_s: int = 120):
    """Poll the llama.cpp server until it is reachable or the timeout elapses."""
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{api_base}/v1/models", timeout=3)
            if r.ok:
                return True
        except Exception as e:
            last_err = e
        time.sleep(2)
    print(f"[ERR ] Server at {api_base} not reachable after {timeout_s}s. Last error: {last_err}", file=sys.stderr)
    return False

def get_model_id(api_base: str, prefer: str | None):
    """Return a valid model id from /v1/models; prefer env if present."""
    try:
        r = requests.get(f"{api_base}/v1/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        if not ids:
            raise RuntimeError("No models returned")
        if prefer and prefer in ids:
            return prefer
        # else pick the first
        return ids[0]
    except Exception as e:
        print(f"[WARN] Failed to query /v1/models ({e}); falling back to env or 'local'", file=sys.stderr)
        return prefer or "local"

def infer_one(src, mt, model_id: str, retries: int = 6):
    """POST ``/v1/chat/completions`` with retry logic for transient errors."""
    payload = {
        "model": model_id,
        "messages": build_messages(src, mt),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "grammar": GRAMMAR_FIELD,
        # You can also try: "stream": False
    }
    backoff = 0.5
    for attempt in range(1, retries+1):
        try:
            r = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, timeout=180)
            if r.status_code in (429, 503):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return sanitize_label(r.json()["choices"][0]["message"]["content"])
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (429, 503) and attempt < retries:
                sleep_s = min(4.0, backoff)
                print(f"[WARN] {code} from server (attempt {attempt}/{retries}) — backing off {sleep_s:.1f}s ...")
                time.sleep(sleep_s)
                backoff *= 2
                continue
            # other HTTP errors or last attempt: raise with context
            body = None
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            raise RuntimeError(f"Server error {code}: {body}") from e
        except requests.RequestException as e:
            # connection/timeout; retry a few times too
            if attempt < retries:
                sleep_s = min(4.0, backoff)
                print(f"[WARN] Request error '{e}' (attempt {attempt}/{retries}) — retrying in {sleep_s:.1f}s ...")
                time.sleep(sleep_s)
                backoff *= 2
                continue
            raise

def main():
    """Run zero-shot evaluation against the configured llama.cpp server."""
    if not wait_for_server(API_BASE, timeout_s=120):
        sys.exit(2)

    model_id = get_model_id(API_BASE, MODEL_ID_ENV)
    print(f"[INFO] Using model id: {model_id}")

    df_full = load_tsv_noheader(DEV_TSV)
    df = df_full.head(EVAL_LIMIT) if EVAL_LIMIT else df_full
    print(f"[INFO] Evaluating {len(df)} row(s) via llama-server at {API_BASE}")

    y_true, y_pred, latencies = [], [], []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Inference", unit="row"):
        t0 = time.perf_counter()
        pred = infer_one(row.src, row.mt, model_id=model_id)
        dt  = time.perf_counter() - t0
        latencies.append(dt)
        y_pred.append(pred)
        y_true.append(row.label)
        print(f"[DEBUG] TRUE={row.label} PRED={pred} | latency={dt:.2f}s")

    map01 = {"ERR":1,"NOT":0}
    yt = [map01.get(y,0) for y in y_true]
    yp = [map01.get(y,0) for y in y_pred]
    mcc = matthews_corrcoef(yt, yp) if len(yt) > 1 else 0.0
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=[1,0], zero_division=0)
    acc = (pd.Series(yt) == pd.Series(yp)).mean()

    print("\n--- Results ---")
    print(f"Subset size: {len(df)}")
    print(f"Accuracy: {acc:.4f}, MCC: {mcc:.4f}")
    print(f"F1-ERR: {f1[0]:.4f}, F1-NOT: {f1[1]:.4f}")
    if latencies:
        print(f"Latency (s): mean={np.mean(latencies):.2f}, max={np.max(latencies):.2f}")

if __name__ == "__main__":
    main()
