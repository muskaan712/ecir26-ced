# Cross-lingual Error Detection (CED)

This repository collects experiments for detecting translation errors in English–German sentences. Models predict whether a machine translation contains an error (**ERR**) or is acceptable (**NOT**).

## Repository Layout

1. **`1_baseline/`** – Initial experiments
   - `1.1_XLM-R/encoder_only.py` – fine-tune `xlm-roberta-large` for sentence-level classification.
   - `1.2_zero_shot/decoder_only.py` – zero-shot evaluation with Mistral‑7B‑Instruct.
   - `1.3_few_shot/decoder_only.py` – few-shot inference using Llama 3 8B with random examples.

2. **`2_scaleup_llm/`** – Scaling with larger language models
   - `2.1_weight_balancing/decoder_only_class_weights.py` – few-shot Llama 3 with oversampled ERR examples.
   - `2.2_cot/cot.py` – chain-of-thought prompting.
   - `2.3_gpt/` – OpenAI GPT models
     - `2.3.1_4o/zeroshot/o4.py` and `2.3.1_4o/fewshot/4o.py` for GPT‑4o.
     - `2.3.2_gpt4.1/zeroshot/gpt4.1.py` and `2.3.2_gpt4.1/fewshot/gpt4.1.py` for GPT‑4.1.
   - `2.4_hybrid/` – combine XLM‑R predictions with GPT/Llama explanations (`gpt.py`, `llama.py`).

3. **`3_modernbert/`** – BERT family comparison
   - `modern_bert.py` – grid search across BERT and ModernBERT models.
   - `mordern_bert_ml.py` – multilingual ModernBERT sentiment variant.

4. **`4_edge_devices/`** – Lightweight models for edge hardware
   - `granite3.3_2B/ed.py`, `llama3.2_3b/ed.py`, `tinyllama/ed.py` – few-shot scripts for small decoder models.

## Data

Scripts expect WMT21 TSV files in `data/wmt21/` containing `ende_majority_train.tsv` and `ende_majority_dev.tsv`.

## Requirements

- Python 3 with PyTorch, `transformers`, `pandas`, `sklearn`, and `tqdm`.
- `huggingface_hub` for model downloads.
- `openai` package and API key for GPT experiments.
- Optional: `tinyllama` fused-kernel package for edge-device scripts.

## Examples

```bash
# XLM-R baseline
python 1_baseline/1.1_XLM-R/encoder_only.py

# GPT-4o zero-shot classification
python 2_scaleup_llm/2.3_gpt/2.3.1_4o/zeroshot/o4.py
```

## Testing

```bash
pytest -q
```

No tests are currently provided; the above command should report that no tests were run.

