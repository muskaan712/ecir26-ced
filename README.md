# Towards Reliable Machine Translation: Scaling LLMs for Critical Error Detection and Safety

This repository gathers the experiments we ran while exploring automatic **critical error detection (CED)** English → German.  It is organized around three phases:

1. **Preliminary baselines** – quick checks with zero-shot and few-shot prompting as well as encoder-only models.
2. **Decision strategies** – prompt-engineering and majority-voting variants that combine multiple model runs.
3. **Fine-tuning** – instruction-tuned or fully fine-tuned large language models specialized for the task.

The code is intentionally lightweight: each experiment lives in its own directory and can be run as a stand-alone script after configuring credentials and data paths.

## Repository layout

```
.
├── 1_preliminary/
│   ├── encoder/           # PyTorch + Hugging Face experiments for encoder-only classifiers
│   ├── fewshot/           # Few-shot prompting scripts for multiple model families
│   └── zeroshot/          # Zero-shot prompting scripts for the same model families
├── 2_strategies/
│   ├── majority_voting/   # Majority-vote aggregations of repeated prompts across models
│   └── prompt_tuning/     # Prompt variants tuned for specific model checkpoints
├── 3_finetuning/          # Supervised fine-tuning scripts for selected LLMs
└── README.md
```

Each subfolder contains a set of scripts targeting a specific model (for example, `llama-8b`, `gpt-oss-120b`, or `gpt-4o`).  The file names (`llama.py`, `oss.py`, `4o_fewshot.py`, …) mirror the model and scenario they were designed for.

## Data requirements

All phases rely on three complementary datasets that we used for every task:

- **WMT21 CED task data** – official EN→DE training/dev/test splits.
- **WMT22 updates** – follow-up releases that expand the evaluation coverage.
- **SynCED-2025** – the latest synthetic augmentation of the critical-error corpus.

## Environment and credentials

- Python 3.10+ with the packages listed in the scripts (PyTorch, Transformers, scikit-learn, pandas, tqdm, etc.).  Creating a fresh virtual environment is recommended.
- Access tokens for hosted LLM APIs (OpenAI, Anthropic, etc.) must be provided through environment variables before running the corresponding scripts.  Many prompting scripts fall back to `OPENAI_API_KEY`, `OAPI`, or provider-specific variables.
- GPU access greatly speeds up the encoder baselines and fine-tuning runs, but the code will fall back to CPU if CUDA is unavailable.

A minimal set of shared dependencies can be installed with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # create this file with the packages you need
```

Alternatively, install only the libraries required for the scripts you plan to run.

## Running experiments

1. **Encoder baselines** (`1_preliminary/encoder/encoder_results.py`)
   - Adjust the file paths at the top of the script.
   - Launch training and evaluation for the configured Hugging Face models:
     ```bash
     python 1_preliminary/encoder/encoder_results.py
     ```
   - The script logs metrics such as accuracy, F1, MCC, and confusion matrices for each encoder.

2. **Zero-shot & few-shot prompts** (`1_preliminary/zeroshot/*`, `1_preliminary/fewshot/*`)
   - Supply API keys via environment variables.
   - Run the script for the target model, e.g.:
     ```bash
     python 1_preliminary/zeroshot/4o/4o_fewshot.py
     ```
   - Scripts log class distributions, prompt completions, and evaluation scores.

3. **Strategy experiments** (`2_strategies/`)
   - Build on the prompting baselines by aggregating multiple runs (majority voting) or specialized prompts (prompt tuning).
   - Execute the script for the relevant model family.  Outputs typically include TSV files with predictions plus detailed logs.

4. **Fine-tuning** (`3_finetuning/`)
   - Configure dataset paths and model identifiers in the script.
   - Use standard `python` invocation to launch fine-tuning.  Checkpoints and evaluation metrics are written to the directory specified in the configuration block.

Each script prints progress to stdout and writes a companion log file (for example, `inference_gpt4o_few_shot.log`) so you can track costs, latency, and metrics.

## Tips for extending the project

- Keep experiment-specific configuration (model names, hyperparameters, file paths) at the top of each script.  This makes it easy to reproduce or adjust runs.
- When adding a new model, duplicate the closest script and update the configuration and prompt instructions accordingly.
- Prefer TSV inputs with the exact column order required by the utilities (EN source, DE hypothesis, label).  Helper functions will raise clear errors if formats differ.
