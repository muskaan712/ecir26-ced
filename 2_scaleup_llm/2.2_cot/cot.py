#!/usr/bin/env python3
# few_shot_llama3_8b_cot.py

import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline
)
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix
)
import time
import logging

# Set up logging
logging.basicConfig(filename="inference_cot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = "/home/s13mchop/LLMs/data/wmt21"
DEV_TSV        = os.path.join(DATA_DIR, "ende_majority_dev.tsv")

HF_TOKEN       = os.getenv("HF_TOKEN")  # ensure this is set
MODEL_ID       = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_ROOT     = "modelcache"
CACHE_DIR      = os.path.join(CACHE_ROOT, MODEL_ID.replace("/", "_"))

BATCH_SIZE     = 8
MAX_NEW_TOKENS = 50
TEMPERATURE    = 0.0

COST_PER_1K_TOKENS = 0.002
# ────────────────────────────────────────────────────────────────────────────────

# ─── Few-Shot Chain-of-Thought Examples ─────────────────────────────────────────
FEW_SHOT_EXAMPLES = (
    # ERR example 1
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: LOL yeah good one mate - have the rest of the day off champ\n"
    "DE: LOL ja gut ein Freund - haben Sie den Rest des Tages freien Champion\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. The English phrase offers someone the rest of the day off, using friendly slang 'champ'.  \n"
    "2. The German literal translation 'freien Champion' misinterprets 'champ' as 'Champion' and is unnatural.  \n"
    "Therefore: ERR<|eot_id|>\n"

    # ERR example 2
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: She was also being very very mean at the second that I made it\n"
    "DE: Sie war auch sehr sehr böse in der Sekunde , dass ich es\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. The source indicates emotional tone and precise timing 'at the second that I made it'.  \n"
    "2. The German truncates the idea and awkwardly splits clauses, losing meaning.  \n"
    "Therefore: ERR<|eot_id|>\n"

    # ERR example 3
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: STOP SENDING ME STUPID MESSAGES. I AM FIXING A MISTAKE IN THE ARTICLE.\n"
    "DE: STOP ME STUPID MESSAGES SSEN . Ich FIXE einen Fehler im Artikel .\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. The English uses imperative and all-caps to show frustration and clarity of actions.  \n"
    "2. The German word order is garbled, splitting 'sending' incorrectly and losing tone.  \n"
    "Therefore: ERR<|eot_id|>\n"

    # ERR example 4
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: This is a test error sentence.\n"
    "DE: Das ist ein Testfehlersatz.\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. The German concatenates words unnaturally, 'Testfehlersatz' is not valid.  \n"
    "2. It changes the clarity of the phrase.  \n"
    "Therefore: ERR<|eot_id|>\n"

    # ERR example 5
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: Another error example.\n"
    "DE: Ein weiteres Fehlerbeispiel.\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. The German translation is literal but drops the article nuance, making it awkward.  \n"
    "2. It loses natural flow in German.  \n"
    "Therefore: ERR<|eot_id|>\n"

    # NOT example 1
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: especially since their actions were often on behalf of the establishment they worked for.\n"
    "DE: vor allem , weil ihre Handlungen oft im Namen der Einrichtung waren , für die sie arbeiteten .\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. Both source and translation convey the same meaning clearly.  \n"
    "2. The German phrasing is accurate and idiomatic.  \n"
    "Therefore: NOT<|eot_id|>\n"

    # NOT example 2
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: I totally agree. Their website lists Italian ice as their product. [URL]\n"
    "DE: Ich stimme dem vollkommen zu. Ihre Website listet italienisches Eis als ihr Produkt. [URL]\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. The content about agreement and listing products matches perfectly.  \n"
    "2. Grammar and terminology are correct.  \n"
    "Therefore: NOT<|eot_id|>\n"

    # NOT example 3
    "<|start_header_id|>user<|end_header_id|>\n"
    "EN: Phalin maybe worthy of a section depending on the Thai impact while the IDL one should be relegated as it were.\n"
    "DE: Phalin kann vielleicht einen Abschnitt wert sein, abhängig von der thailändischen Wirkung, während die IDL einem Abschnitt untergeordnet werden sollte.\n"
    "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let’s think step by step:\n"
    "1. The English nuance about section importance is preserved.  \n"
    "2. The conditional structure maps correctly into German.  \n"
    "Therefore: NOT<|eot_id|>\n"
)
# ────────────────────────────────────────────────────────────────────────────────

def download_and_cache_model():
    """Download MODEL_ID into CACHE_DIR once, then reuse."""
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=CACHE_DIR,
            token=HF_TOKEN
        )


def load_data(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["id","src","mt","toklabels","label"],
        dtype={"id": str}
    )
    df["label_id"] = df["label"].map({"ERR": 1, "NOT": 0})
    return df


def make_prompt(src: str, mt: str) -> str:
    # Combine system prompt, few-shot CoT examples, and user query
    system_prompt = (
        "You are a precise translation evaluator. For each English source (EN) and German translation (DE), "
        "first think through whether the translation preserves all key meanings, point out any shifts or omissions in numbered steps, "
        "and only then reply with ‘ERR’ (clear error) or ‘NOT’ (accurate/minor imperfections). "
        "Begin your reasoning with “Let’s think step by step:” and conclude with “Therefore: ERR” or “Therefore: NOT”."
    )
    user_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"EN: {src.strip()}\nDE: {mt.strip()}\n"
        "Reply ONLY with step-by-step reasoning and then ERR or NOT.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>\n"
        f"{FEW_SHOT_EXAMPLES}"
        f"{user_prompt}"
    )


def extract_text(out):
    if isinstance(out, dict) and "generated_text" in out:
        return out["generated_text"]
    if isinstance(out, str):
        return out
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            return first.get("generated_text", "")
        if isinstance(first, str):
            return first
    return ""


def main():
    # 1) Cache model locally
    download_and_cache_model()

    # 2) Load dev data & build prompts
    df = load_data(DEV_TSV)
    prompts = [make_prompt(r.src, r.mt) for _, r in df.iterrows()]

    # 3) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        CACHE_DIR, use_fast=True, local_files_only=True
    )
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = AutoModelForCausalLM.from_pretrained(
        CACHE_DIR,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 4) Create generation pipeline
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,
        return_full_text=False
    )

    # 5) Inference + logging + metrics
    gen_texts, preds = [], []
    total_tokens, start_time = 0, time.time()

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="CoT few-shot (Llama 3 8B)"):
        batch = prompts[i:i+BATCH_SIZE]
        batch_start = time.time()
        outputs = pipe(batch)
        batch_end = time.time()
        logging.info(f"Batch {i//BATCH_SIZE+1}: size={len(batch)}, latency={batch_end-batch_start:.3f}s")

        for out, prompt in zip(outputs, batch):
            gen = extract_text(out).strip().upper()
            gen_texts.append(gen)
            in_tokens = len(tokenizer.encode(prompt))
            out_tokens = len(tokenizer.encode(gen))
            total_tokens += in_tokens + out_tokens
            pred = 1 if 'ERR' in gen else 0
            preds.append(pred)

    end_time = time.time()
    logging.info(f"Total latency: {end_time-start_time:.3f}s, tokens: {total_tokens}")
    cost_estimate = (total_tokens/1000)*COST_PER_1K_TOKENS
    logging.info(f"Estimated cost: ${cost_estimate:.4f}")

    # Print preview
    print("\nFirst 10 (generated → true/pred):\n")
    for i in range(10):
        gt = gen_texts[i] or "<EMPTY>"
        true = "ERR" if df.loc[i,'label_id']==1 else "NOT"
        pred = "ERR" if preds[i]==1 else "NOT"
        print(f"#{i+1:2d} Generated: {gt!r}\n    True / Pred: {true} / {pred}\n")

    # Compute metrics
    labels = df["label_id"].tolist()
    mcc = matthews_corrcoef(labels, preds)
    prf = precision_recall_fscore_support(labels, preds, labels=[1,0], zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[1,0])

    print(f"\nMCC   : {mcc:.4f}")
    print(f"F1-ERR: {prf[2][0]:.4f}")
    print(f"F1-NOT: {prf[2][1]:.4f}\n")
    print("Confusion Matrix (rows=true │ cols=pred)")
    print("      ERR   NOT")
    print(f"ERR {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"NOT {cm[1,0]:6d} {cm[1,1]:6d}")

if __name__ == "__main__":
    main()
