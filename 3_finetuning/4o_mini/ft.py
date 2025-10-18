#!/usr/bin/env python3
"""Utility for fine-tuning GPT-4o mini on a prepared JSONL dataset.

The script performs three tasks:

1. Inspect the input dataset for sanity checks (counts and label balance).
2. Upload the JSONL to OpenAI and launch a fine-tuning job.
3. Poll the job for new events and stream them to stdout until completion.

Environment variables ``OPENAI_API_KEY`` or ``OAPI`` must be set. Adjust the
configuration constants below to match your workspace before running.
"""

# ft_4omini_full_launch.py
# Use 100% of an existing JSONL to fine-tune 4o-mini (EPOCHS configurable),
# and stream events live (no blank waits).

import os, json, time, sys

# ── Edit these ────────────────────────────────────────────────────────────────
SRC_JSONL    = "/path/to/datasets/train.jsonl"          # existing full JSONL
BASE_MODEL   = "gpt-4o-mini-2024-07-18"
MODEL_SUFFIX = "synced-label-4o-mini"  # optional; None=omit
EPOCHS       = 2              # adjust if you need to control cost
SLEEP_SECS   = 0              # 0 = no wait between polls
# Uses OPENAI_API_KEY (or OAPI) from env
# ──────────────────────────────────────────────────────────────────────────────

try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
p = lambda *a, **k: print(*a, **{**k, "flush": True})

def inspect_dataset(in_path):
    """Count lines and label distribution for visibility."""
    if not os.path.exists(in_path):
        raise SystemExit(f"[ERROR] Source JSONL not found: {in_path}")
    total = err = not_ = 0
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            total += 1
            try:
                obj = json.loads(line)
                label = obj["messages"][-1]["content"].strip().upper()
                if label == "ERR":
                    err += 1
                elif label == "NOT":
                    not_ += 1
            except Exception:
                # If any odd line appears, just count it toward total
                pass
    p(f"[INFO] Dataset lines      : {total}")
    p(f"[INFO] Label distribution : ERR={err}  NOT={not_}")
    return total

def launch_and_stream(training_jsonl):
    """Create a fine-tuning job and stream status updates until it ends."""
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
            if e.id in seen: 
                continue
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
    """Validate the dataset, launch a job, and stream progress logs."""
    p("[START] Using FULL dataset (no sampling)")
    inspect_dataset(SRC_JSONL)
    p("[FINISH] Dataset inspection")
    launch_and_stream(SRC_JSONL)

if __name__ == "__main__":
    main()
