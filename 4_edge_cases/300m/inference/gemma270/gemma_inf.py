#!/usr/bin/env python3
# gemma_ced_inference_llscore_plus.py
# Deterministic CED (EN→DE) with Gemma using 2-label LL scoring + improvements:
# - Verbalizer sets + log-sum-exp over tokenization variants
# - Optional contextual calibration (label prior)
# - Δ sweep to pick best margin on DEV (max MCC)
# - Stores margins so you can re-threshold cheaply

import os, re, logging, math
from typing import Tuple, Optional, List, Dict
from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

# ───────────────────────────── Config ─────────────────────────────
DATA_DIR    = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV     = os.path.join(DATA_DIR, "ende_majority_dev.tsv")

MODEL_ID    = "google/gemma-3-270m-it"
HF_TOKEN    = os.getenv("HF_TOKEN")  # optional for gated models

DESC             = "Gemma CED eval (LL scoring + calib + sweep)"
LOG_FILE         = "inference_gemma_llscore_plus.log"

# If None → we sweep Δ; if a float → we use it directly (no sweep)
DELTA_LOGPROB    = None
DELTA_SWEEP      = [x/20.0 for x in range(-80, 81)]  # -4.0 … +4.0 step 0.05

# Contextual calibration: subtract a label prior measured on a neutral prompt
USE_CONTEXT_CAL  = True

# Few-shot anchors (optional). Keep short and balanced; set FEW_SHOTS=[] to disable.
FEW_SHOTS: List[Tuple[str,str,str]] = [
    # (src_en, mt_de, label) — very terse, unambiguous pairs
    ("The patient denies any allergies.", "Der Patient bestreitet Allergien.", "NOT"),
    ("He didn’t take the medicine.", "Er nahm die Medizin.", "ERR"),
]

LABELS = {"ERR","NOT"}
VERBALIZERS: Dict[str, List[str]] = {
    # Keep concise; capital forms usually align with instruction heads
    "ERR": ["ERR", "YES"],
    "NOT": ["NOT", "NO"],
}

# ──────────────────────────── Logging ────────────────────────────
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

# ───────────────────────────── Helpers ────────────────────────────
def safe_len(s: str, max_chars: int = 10000) -> str:
    if s is None: return ""
    s = str(s)
    if len(s) <= max_chars: return s
    half = max_chars // 2
    return s[:half] + "\n...[TRUNCATED]...\n" + s[-half:]

def clean_double_eq(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if "==" not in text:
        return text.strip()
    parts = [p.strip() for p in text.split("==") if p.strip()]
    if not parts: return ""
    return max(parts, key=len)

def is_bracket_vec(series: pd.Series) -> bool:
    patt = re.compile(r"^\s*\[\s*([0-9]+(\s*,\s*[0-9]+)*)\s*\]\s*$")
    n = len(series)
    if n == 0: return False
    hits = sum(bool(patt.match(str(x))) for x in series)
    return hits >= max(3, int(0.8 * n))

def is_id_like(series: pd.Series) -> bool:
    patt = re.compile(r"^\s*[0-9]{1,8}\s*$")
    n = len(series)
    if n == 0: return False
    hits = sum(bool(patt.match(str(x))) for x in series)
    return hits >= max(3, int(0.8 * n))

def is_labelish(series: pd.Series) -> bool:
    def map_one(x: str) -> Optional[str]:
        s = str(x).strip().upper()
        if s in LABELS: return s
        if s in {"E","ERROR","YES","Y","TRUE","T","1"}: return "ERR"
        if s in {"N","NO","FALSE","F","0"}: return "NOT"
        if s.startswith("E"): return "ERR"
        return None
    n = len(series)
    if n == 0: return False
    mapped = [map_one(v) for v in series]
    hits = sum(1 for m in mapped if m is not None)
    return hits >= max(3, int(0.7 * n)) and series.astype(str).str.len().median() <= 5

def coerce_label(x: str) -> str:
    if x is None: return "NOT"
    s = str(x).strip().upper()
    if s in LABELS: return s
    if s in {"E","ERROR","YES","Y","TRUE","T","1"}: return "ERR"
    if s in {"N","NO","FALSE","F","0"}: return "NOT"
    if s.startswith("E"): return "ERR"
    return "NOT"

# ───────────────────── Column detection / reading ─────────────────
def read_tsv_flex(path: str) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, na_filter=False)

    if df.shape[1] == 2:
        df.columns = ["src", "mt"]
        df["src"] = df["src"].map(clean_double_eq)
        df["mt"]  = df["mt"].map(clean_double_eq)
        return df, "src", "mt", None

    if df.shape[1] == 3:
        df.columns = ["src", "mt", "label"]
        df["src"] = df["src"].map(clean_double_eq)
        df["mt"]  = df["mt"].map(clean_double_eq)
        return df, "src", "mt", "label"

    if df.shape[1] == 5:
        df.columns = ["id_col", "src_raw", "mt_raw", "vec_col", "label_raw"]
        out = pd.DataFrame({
            "src": df["src_raw"].map(clean_double_eq),
            "mt":  df["mt_raw"].map(clean_double_eq),
            "label": df["label_raw"]
        })
        return out, "src", "mt", "label"

    keep_cols = []
    for i in range(df.shape[1]):
        col = df[i]
        if is_bracket_vec(col) or is_id_like(col):
            continue
        keep_cols.append(i)
    if len(keep_cols) < 2:
        keep_cols = list(range(df.shape[1]))
    sub = df[keep_cols].copy()

    label_col_idx = None
    for i in sub.columns:
        if is_labelish(sub[i]):
            label_col_idx = i
            break

    lens = [(i, sub[i].astype(str).str.len().median()) for i in sub.columns if i != label_col_idx]
    lens.sort(key=lambda x: x[1], reverse=True)
    if len(lens) < 2:
        raise ValueError(f"Could not find two text columns in TSV (columns kept: {list(sub.columns)})")
    src_idx, mt_idx = lens[0][0], lens[1][0]

    out = pd.DataFrame({
        "src": sub[src_idx].map(clean_double_eq),
        "mt":  sub[mt_idx].map(clean_double_eq),
    })
    label_name: Optional[str] = None
    if label_col_idx is not None:
        out["label"] = sub[label_col_idx]
        label_name = "label"
    return out, "src", "mt", label_name

# ───────────────────────── Model & Prompt ────────────────────────
def load_gemma(model_id: str, hf_token: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, torch_dtype="auto", device_map="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model

def build_messages(src: str, mt: str):
    sys_msg  = "Binary classifier for MT critical errors. Reply with exactly one token: ERR or NOT. If uncertain, reply NOT. No other text."
    parts = []
    # optional few-shot anchors
    for s, m, y in FEW_SHOTS:
        parts.append(f"EN: {s}\nDE: {m}\nLabel: {y}")
    parts.append(f"EN: {safe_len(src)}\nDE: {safe_len(mt)}\nLabel:")
    user_msg = "\n\n".join(parts)
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg},
    ]

# ─────────── log-likelihood scoring helpers (deterministic) ───────────
def _encode_variants(tokenizer, s: str, device) -> List[torch.Tensor]:
    variants = [s, " " + s, s + "\n", " " + s + "\n"]
    outs = []
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) > 0:
            outs.append(torch.tensor(ids, dtype=torch.long, device=device))
    uniq, seen = [], set()
    for t in outs:
        key = tuple(t.tolist())
        if key not in seen:
            uniq.append(t); seen.add(key)
    return uniq

def score_seq_logp(prefix_ids: torch.Tensor, model, label_ids: torch.Tensor) -> float:
    full_ids = torch.cat([prefix_ids, label_ids.unsqueeze(0)], dim=1)  # (1, L+K)
    with torch.no_grad():
        out = model(input_ids=full_ids)
        logits = out.logits[0]  # (L+K, V)
        log_probs = torch.log_softmax(logits, dim=-1)
    L = prefix_ids.shape[1]
    K = label_ids.shape[0]
    tok_logps = []
    for k in range(K):
        tok_id = int(label_ids[k].item())
        tok_logps.append(log_probs[L + k - 1, tok_id])
    return float(torch.stack(tok_logps).sum().item())

def group_logprob(prefix_ids: torch.Tensor, tokenizer, model, verbalizers: List[str]) -> float:
    scores = []
    for word in verbalizers:
        for ids in _encode_variants(tokenizer, word, model.device):
            scores.append(score_seq_logp(prefix_ids, model, ids))
    # log-sum-exp over all variants to aggregate probability mass for a class
    t = torch.tensor(scores, device=model.device)
    return float(torch.logsumexp(t, dim=0).item())

def class_margin(tokenizer, model, messages, calib_offsets: Optional[Dict[str,float]]=None) -> float:
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    prefix_ids = inputs  # (1, L)

    err_logp = group_logprob(prefix_ids, tokenizer, model, VERBALIZERS["ERR"])
    not_logp = group_logprob(prefix_ids, tokenizer, model, VERBALIZERS["NOT"])

    if calib_offsets:
        err_logp -= calib_offsets.get("ERR", 0.0)
        not_logp -= calib_offsets.get("NOT", 0.0)

    return err_logp - not_logp  # positive → ERR

def compute_calibration_offsets(tokenizer, model) -> Dict[str, float]:
    # Neutral prompt: keeps instruction & formatting but with trivial content
    neutral = build_messages(src=".", mt=".")
    inputs = tokenizer.apply_chat_template(neutral, add_generation_prompt=True, return_tensors="pt").to(model.device)
    prefix_ids = inputs
    err = group_logprob(prefix_ids, tokenizer, model, VERBALIZERS["ERR"])
    not_ = group_logprob(prefix_ids, tokenizer, model, VERBALIZERS["NOT"])
    return {"ERR": err, "NOT": not_}

# ───────────────────────── Evaluation ────────────────────────────
def evaluate(y_true: List[str], margins: List[float], delta: float):
    y_pred = ["ERR" if m >= delta else "NOT" for m in margins]
    labels = ["ERR", "NOT"]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "p_err": float(prec[0]), "r_err": float(rec[0]), "f_err": float(f1[0]), "support_err": int(sup[0]),
        "p_not": float(prec[1]), "r_not": float(rec[1]), "f_not": float(f1[1]), "support_not": int(sup[1]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cm": confusion_matrix(y_true, y_pred, labels=labels),
        "delta": float(delta),
    }
    return metrics, y_pred

# ───────────────────────────── Main ─────────────────────────────
def main():
    # 1) Data
    df, src_col, mt_col, label_col = read_tsv_flex(DEV_TSV)

    # 2) Model
    tokenizer, model = load_gemma(MODEL_ID, HF_TOKEN)

    # 3) Calibration offset (optional)
    calib = compute_calibration_offsets(tokenizer, model) if USE_CONTEXT_CAL else None

    # 4) Score margins once
    margins, y_true = [], []
    it = tqdm(df.itertuples(index=False), total=len(df), desc=DESC)
    has_labels = label_col is not None
    for row in it:
        rowd = row._asdict()
        src, mt = rowd[src_col], rowd[mt_col]
        if has_labels:
            y_true.append(coerce_label(rowd[label_col]))
        try:
            m = class_margin(tokenizer, model, build_messages(src, mt), calib_offsets=calib)
        except Exception as e:
            logging.exception(f"Scoring failed: {e}")
            m = -999.0  # extreme NOT
        margins.append(m)

    # 5) Threshold selection & report
    if has_labels and len(y_true) == len(margins) and len(y_true) > 0:
        if DELTA_LOGPROB is None:
            best = None
            for d in DELTA_SWEEP:
                met, _ = evaluate(y_true, margins, d)
                if (best is None) or (met["mcc"] > best["mcc"]):
                    best = met
            delta = best["delta"]
            metrics, y_pred = evaluate(y_true, margins, delta)
            print(f"\n[Δ-sweep] Best Δ = {delta:.4f} | MCC(dev) = {metrics['mcc']:.4f}")
        else:
            metrics, y_pred = evaluate(y_true, margins, DELTA_LOGPROB)
            print(f"\n[Δ-fixed] Δ = {DELTA_LOGPROB:.4f} | MCC(dev) = {metrics['mcc']:.4f}")

        cm = metrics["cm"]
        print(f"F1-ERR: {metrics['f_err']:.4f} | F1-NOT: {metrics['f_not']:.4f}")
        print("\nConfusion Matrix (rows=true │ cols=pred)")
        print("      ERR   NOT")
        print(f"ERR  {cm[0,0]:5d} {cm[0,1]:5d}")
        print(f"NOT  {cm[1,0]:5d} {cm[1,1]:5d}")

        acc = sum(("ERR" if m >= metrics["delta"] else "NOT") == t for m, t in zip(margins, y_true)) / len(y_true)
        cnt = Counter(y_true)
        print(f"\nAccuracy: {acc:.4f}")
        print(f"Total examples: {len(y_true)}")
        print(f"ERR examples: {cnt['ERR']}")
        print(f"NOT examples: {cnt['NOT']}")
    else:
        # no labels → just count predictions using a conservative Δ=0.7 (or your choice)
        delta = 0.7 if DELTA_LOGPROB is None else DELTA_LOGPROB
        preds = ["ERR" if m >= delta else "NOT" for m in margins]
        cnt_pred = Counter(preds)
        print("\n(No labels detected) Completed predictions.")
        print(f"Δ used: {delta:.2f}")
        print(f"Total examples: {len(preds)}")
        print(f"Pred counts → ERR: {cnt_pred['ERR']}, NOT: {cnt_pred['NOT']}")

if __name__ == "__main__":
    main()
