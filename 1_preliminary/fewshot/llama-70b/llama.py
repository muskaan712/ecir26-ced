#!/usr/bin/env python3
"""Few-shot Critical Error Detection via llama.cpp's llama-server."""
# Few-shot CED (EN→DE) via llama-server (GGUF, GPU via llama.cpp server)
# - Adds 5 ERR + 3 NOT few-shot demos sampled from TRAIN (ERR first, then NOT)
# - Supports TSV **or** split files: *.src, *.mt, *.label (e.g., dev.src/dev.mt/dev.label)
# - Strict grammar ("ERR" | "NOT"), 1-token output
# - tqdm with ETA + latency profiling
# - Robust: auto-detect model id + retry on 503/429

import os, time, requests, sys, random, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

# ===================== CONFIG =====================
API_BASE      = os.environ.get("API_BASE", "http://127.0.0.1:8811")
MODEL_ID_ENV  = os.environ.get("MODEL_ID", "").strip()

# >>> Point these to EITHER a TSV file OR a directory with *.src/*.mt/*.label
TRAIN_PATH = "/path/to/train_data"   # e.g., contains train.src, train.mt, train.label
DEV_PATH   = "/path/to/dev_data"     # e.g., contains dev.src, dev.mt, dev.label
# (You can also set them to TSV files: ".../ende_wmt22_train.tsv", ".../ende_wmt22_dev.tsv")

EVAL_LIMIT      = None      # set to None or 0 for full dataset
FEWSHOT_ERR_N   = 5
FEWSHOT_NOT_N   = 3
FEWSHOT_SEED    = 42        # deterministic sampling

MAX_NEW_TOKENS  = 1
TEMPERATURE     = 0.0
TOP_P           = 1.0

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

# ---------- Loaders (TSV or split files) ----------
_LABEL_MAP = {
    "ERR":"ERR", "NOT":"NOT",
    "BAD":"ERR", "OK":"NOT",
    "ERROR":"ERR", "CORRECT":"NOT",
}

def _normalize_label(x: str) -> str:
    t = (x or "").strip().upper()
    return _LABEL_MAP.get(t, t if t in ("ERR","NOT") else "ERR")

def load_tsv_noheader(path):
    """Load a TSV dataset without headers into a normalized dataframe."""
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
    with open(fp, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def load_split_dir(dir_path: str) -> pd.DataFrame:
    """Load dataset from a directory containing *.src, *.mt, *.label (any prefix)."""
    src_files   = sorted(glob.glob(os.path.join(dir_path, "*.src")))
    mt_files    = sorted(glob.glob(os.path.join(dir_path, "*.mt")))
    label_files = sorted(glob.glob(os.path.join(dir_path, "*.label")))
    if not (src_files and mt_files and label_files):
        raise FileNotFoundError(f"Missing split files in {dir_path}. Need *.src, *.mt, *.label")

    # Pick the first match of each kind (supports dev.* or train.*)
    src_fp, mt_fp, label_fp = src_files[0], mt_files[0], label_files[0]

    src = _read_lines(src_fp)
    mt  = _read_lines(mt_fp)
    lb  = _read_lines(label_fp)
    n = min(len(src), len(mt), len(lb))
    if not (len(src) == len(mt) == len(lb)):
        print(f"[WARN] Length mismatch in split files: src={len(src)} mt={len(mt)} label={len(lb)}. Truncating to {n}.", file=sys.stderr)

    df = pd.DataFrame({"src": src[:n], "mt": mt[:n], "label": [ _normalize_label(x) for x in lb[:n] ]}, dtype=str)
    return df

def load_any_dataset(path_or_dir: str) -> pd.DataFrame:
    """Auto-detect loader: TSV file or split directory."""
    p = (path_or_dir or "").strip()
    if not p:
        raise ValueError("Empty dataset path")
    if os.path.isdir(p):
        return load_split_dir(p)
    # treat as file
    ext = os.path.splitext(p)[1].lower()
    if ext in (".tsv", ".txt"):
        return load_tsv_noheader(p)
    # if user points to one of the split files directly, use its directory
    if ext in (".src", ".mt", ".label"):
        return load_split_dir(os.path.dirname(p))
    # default: try TSV parser
    return load_tsv_noheader(p)

# ---------- Few-shot sampling & message building ----------
def sanitize_label(text: str) -> str:
    """Reduce arbitrary responses to a deterministic ERR/NOT decision."""
    t = text.strip().upper()
    if "ERR" in t and "NOT" in t:
        return "ERR" if t.index("ERR") < t.index("NOT") else "NOT"
    if "ERR" in t: return "ERR"
    if "NOT" in t: return "NOT"
    return "ERR"

def wait_for_server(api_base: str, timeout_s: int = 120):
    """Poll the llama-server until it responds or a timeout is reached."""
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
        return ids[0]
    except Exception as e:
        print(f"[WARN] Failed to query /v1/models ({e}); falling back to env or 'local'", file=sys.stderr)
        return prefer or "local"

def sample_fewshot(df_train: pd.DataFrame, n_err=5, n_not=3, seed=42):
    """Return list of demo tuples: [(src, mt, label), ...] in order: all ERR, then NOT."""
    rnd = random.Random(seed)
    err_rows = df_train[df_train["label"] == "ERR"].copy()
    not_rows = df_train[df_train["label"] == "NOT"].copy()

    err_idx = list(err_rows.index); not_idx = list(not_rows.index)
    rnd.shuffle(err_idx); rnd.shuffle(not_idx)

    err_pick = err_rows.loc[err_idx[:min(n_err, len(err_idx))]]
    not_pick = not_rows.loc[not_idx[:min(n_not, len(not_idx))]]

    demos = []
    for r in err_pick.itertuples(index=False):
        demos.append((r.src, r.mt, "ERR"))
    for r in not_pick.itertuples(index=False):
        demos.append((r.src, r.mt, "NOT"))
    return demos

def build_messages_with_fewshot(src: str, mt: str, demos):
    """
    Compose messages as:
      system: SYSTEM_PROMPT
      user/assistant pairs for each demo
      user: current EN/DE
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for (d_src, d_mt, d_label) in demos:
        messages.append({"role": "user",      "content": f"EN: {d_src.strip()}\nDE: {d_mt.strip()}"})
        messages.append({"role": "assistant", "content": d_label})
    messages.append({"role": "user", "content": f"EN: {src.strip()}\nDE: {mt.strip()}"})
    return messages

def infer_one(src, mt, model_id: str, demos, retries: int = 6):
    """POST /v1/chat/completions with retry on 503/429."""
    payload = {
        "model": model_id,
        "messages": build_messages_with_fewshot(src, mt, demos),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "grammar": GRAMMAR_FIELD,
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
                time.sleep(sleep_s); backoff *= 2; continue
            body = None
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            raise RuntimeError(f"Server error {code}: {body}") from e
        except requests.RequestException as e:
            if attempt < retries:
                sleep_s = min(4.0, backoff)
                print(f"[WARN] Request error '{e}' (attempt {attempt}/{retries}) — retrying in {sleep_s:.1f}s ...")
                time.sleep(sleep_s); backoff *= 2; continue
            raise

def main():
    """Run the end-to-end evaluation loop against llama-server."""
    if not wait_for_server(API_BASE, timeout_s=120):
        sys.exit(2)

    model_id = get_model_id(API_BASE, MODEL_ID_ENV)
    print(f"[INFO] Using model id: {model_id}")

    # Load DEV/EVAL (auto-detect TSV vs split dir)
    df_full = load_any_dataset(DEV_PATH)
    df = df_full.head(EVAL_LIMIT) if EVAL_LIMIT else df_full
    print(f"[INFO] Evaluating {len(df)} row(s) via llama-server at {API_BASE}")

    # Load TRAIN for few-shot (fallback to DEV if not provided/failed)
    train_path = (TRAIN_PATH or "").strip()
    if not train_path:
        print("[WARN] TRAIN_PATH is empty. Falling back to DEV_PATH for few-shot sampling.", file=sys.stderr)
        train_path = DEV_PATH

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

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Inference", unit="row"):
        t0 = time.perf_counter()
        pred = infer_one(row.src, row.mt, model_id=model_id, demos=demos)
        dt  = time.perf_counter() - t0
        latencies.append(dt)
        y_pred.append(pred)
        y_true.append(row.label)
        print(f"[DEBUG] TRUE={row.label} PRED={pred} | latency={dt:.2f}s")

    map01 = {"ERR":1, "NOT":0}
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
