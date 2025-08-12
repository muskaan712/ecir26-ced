#!/usr/bin/env python3
# ft_4omini_half_launch.py
# Make a 50% stratified sample from an existing JSONL, fine-tune 4o-mini (1 epoch),
# and stream events live (no blank waits).

import os, json, random, time, sys

# ── Edit these ────────────────────────────────────────────────────────────────
SRC_JSONL   = "/home/s13mchop/LLMs/data/wmt21/ft/training_set.jsonl"   # existing full JSONL
OUT_JSONL   = "/home/s13mchop/LLMs/data/wmt21/ft/training_set_50.jsonl"
BASE_MODEL  = "gpt-4o-mini-2024-07-18"
MODEL_SUFFIX = "ced-label-only-50pct-v1"
EPOCHS       = 3              # keep cheap to pass hard limit precheck
SLEEP_SECS   = 0              # 0 = no wait between polls
# Uses OPENAI_API_KEY (or OAPI) from env
# ──────────────────────────────────────────────────────────────────────────────

try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
p = lambda *a, **k: print(*a, **{**k, "flush": True})

def stratified_half(in_path, out_path, seed=13):
    p("[START] Sampling 50% (stratified)")
    errs, nots = [], []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            # Assistant label is last message
            label = obj["messages"][-1]["content"].strip().upper()
            (errs if label == "ERR" else nots).append(line)

    random.seed(seed)
    k_err = max(1, round(len(errs) * 0.5))
    k_not = max(1, round(len(nots) * 0.5))
    sample = random.sample(errs, k_err) + random.sample(nots, k_not)
    random.shuffle(sample)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(sample)

    p(f"[FINISH] Sampling — ERR {k_err}/{len(errs)}, NOT {k_not}/{len(nots)}")
    p(f"[RESULT] Wrote {len(sample)} lines → {out_path}")
    return len(sample)

def launch_and_stream(training_jsonl):
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OAPI")
    if not key:
        raise SystemExit("[ERROR] Missing OPENAI_API_KEY (or OAPI).")

    client = OpenAI(api_key=key)

    p("[START] Upload training file")
    with open(training_jsonl, "rb") as f:
        tr = client.files.create(file=f, purpose="fine-tune")
    p(f"[FINISH] Upload — file_id={tr.id}")

    p("[START] Create fine-tuning job")
    job = client.fine_tuning.jobs.create(
        model=BASE_MODEL,
        training_file=tr.id,
        suffix=MODEL_SUFFIX or None,
        hyperparameters={"n_epochs": EPOCHS}
    )
    p(f"[FINISH] Create job — job_id={job.id} status={job.status}")

    p("[START] Stream events")
    seen = set()
    ts = lambda s: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(s))
    while True:
        ev = client.fine_tuning.jobs.list_events(job.id, limit=200)
        for e in reversed(ev.data):
            if e.id in seen: continue
            seen.add(e.id)
            lvl = (e.level or "info").upper()
            p(f"[EVENT {ts(e.created_at)}] {lvl:<5} {e.message or ''}")

        j = client.fine_tuning.jobs.retrieve(job.id)
        if j.status in ("succeeded","failed","cancelled"):
            p(f"[FINISH] Stream events — final_status={j.status}")
            p(f"[RESULT] fine_tuned_model={getattr(j,'fine_tuned_model', None)}")
            break

        if SLEEP_SECS > 0:
            time.sleep(SLEEP_SECS)

def main():
    if not os.path.exists(SRC_JSONL):
        raise SystemExit(f"[ERROR] Source JSONL not found: {SRC_JSONL}")

    stratified_half(SRC_JSONL, OUT_JSONL)
    launch_and_stream(OUT_JSONL)

if __name__ == "__main__":
    main()
