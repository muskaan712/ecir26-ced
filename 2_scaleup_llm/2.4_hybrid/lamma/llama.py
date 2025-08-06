import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextGenerationPipeline
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download

# ---------- CONFIGURATION ----------
DATA_DIR = "/home/s13mchop/LLMs/data/wmt21"
TRAIN_TSV = os.path.join(DATA_DIR, "ende_majority_train.tsv")
DEV_TSV = os.path.join(DATA_DIR, "ende_majority_dev.tsv")
MODEL_NAME = "xlm-roberta-large"
BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-6
MAX_LENGTH = 256
DEVICE = torch.device("cuda")
ROW_LIMIT = 1000

# Llama config
DECODER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_ROOT = "/home/s13mchop/LLMs/ecir-ced/2_scaleup_llm/2.4_hybrid/gpt/modelcache"
CACHE_DECODER = os.path.join(CACHE_ROOT, DECODER_MODEL.replace('/', '_'))
EXPL_TOKENS = 50
TEMPERATURE = 0.0
FEW_SHOT_ERR = 5
FEW_SHOT_NOT = 3

# ---------- UTILS ----------
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

class CEDDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_length=256, limit=None):
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                         names=['id','src','mt','toklabels','label'], dtype={0:int,1:str,2:str,4:str})
        if limit:
            df = df.head(limit)
        self.src = df['src'].astype(str).tolist()
        self.mt = df['mt'].astype(str).tolist()
        self.labels = df['label'].map({'ERR':1, 'NOT':0}).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        src_text = clean_text(self.src[idx])
        mt_text = clean_text(self.mt[idx])
        inputs = self.tokenizer(src_text, mt_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['src'] = src_text
        item['mt'] = mt_text
        return item

class CEDModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h0 = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(h0)
        return logits

def select_examples_from_training(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['id','src','mt','toklabels','label'])
    err = df[df.label=='ERR'].sample(FEW_SHOT_ERR, random_state=42)
    not_ = df[df.label=='NOT'].sample(FEW_SHOT_NOT, random_state=42)
    examples = []
    for _, row in pd.concat([err, not_]).iterrows():
        lbl = row.label
        expl = ('Major meaning error or omission detected.' if lbl=='ERR' else 'Meaning preserved; only minor stylistic issues.')
        examples.append({'src': row.src, 'mt': row.mt, 'label': lbl, 'explanation': expl})
    return examples

SYSTEM_PROMPT = ("You are a professional translation quality analyst. Given EN, DE, and a label, "
    "explain in ≤ 50 words why the translation is erroneous (ERR) or acceptable (NOT).")
def build_decoder_prompt(src, mt, label, few_shot):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + SYSTEM_PROMPT + "<|eot_id|>\n"
    for ex in few_shot:
        prompt += (f"<|start_header_id|>user<|end_header_id|>\n"
                   f"EN: {ex['src']}\nDE: {ex['mt']}\nPREDICTED: {ex['label']}<|eot_id|>\n"
                   f"<|start_header_id|>assistant<|end_header_id|>\n{ex['explanation']}<|eot_id|>\n")
    prompt += (f"<|start_header_id|>user<|end_header_id|>\nEN: {src}\nDE: {mt}\nPREDICTED: {label}<|eot_id|>\n"
               f"<|start_header_id|>assistant<|end_header_id|>\n")
    return prompt

def generate_explanations(pipe, src_list, mt_list, pred_list, few_shot):
    explanations = []
    for src, mt, pred in tqdm(zip(src_list, mt_list, pred_list), total=len(src_list), desc="Generating Explanations"):
        label_str = 'ERR' if pred == 1 else 'NOT'
        prompt = build_decoder_prompt(src, mt, label_str, few_shot)
        out = pipe(prompt)
        explanations.append(out[0]['generated_text'].strip())
    return explanations

# ---------- MAIN ----------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = CEDDataset(TRAIN_TSV, tokenizer, MAX_LENGTH)
    dev_ds = CEDDataset(DEV_TSV, tokenizer, MAX_LENGTH, limit=ROW_LIMIT)

    labels = train_ds.labels
    class_counts = [labels.count(0), labels.count(1)]
    total = sum(class_counts)
    weights = [total/(2*c) for c in class_counts]
    sample_weights = [weights[label] for label in labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)

    model = CEDModel(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-6)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS+1):  # Training
        model.train()
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].long().to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    model.eval()

    # Inference (first 10 only)
    src_list, mt_list, pred_list = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).tolist()
            src_list.extend(batch['src'])
            mt_list.extend(batch['mt'])
            pred_list.extend(preds)
    src_list, mt_list, pred_list = src_list[:10], mt_list[:10], pred_list[:10]

    # Llama decoder for explainability
    snapshot_download(repo_id=DECODER_MODEL, local_dir=CACHE_DECODER, token=os.getenv('HF_TOKEN'))
    dec_tokenizer = AutoTokenizer.from_pretrained(CACHE_DECODER, use_fast=True, local_files_only=True)
    dec_tokenizer.add_special_tokens({'pad_token': dec_tokenizer.eos_token})
    decoder = AutoModelForCausalLM.from_pretrained(CACHE_DECODER, local_files_only=True, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
    pipe = TextGenerationPipeline(model=decoder, tokenizer=dec_tokenizer, max_new_tokens=EXPL_TOKENS, temperature=TEMPERATURE, do_sample=False, return_full_text=False)

    few_shot = select_examples_from_training(TRAIN_TSV)
    explanations = generate_explanations(pipe, src_list, mt_list, pred_list, few_shot)

    print("\nFirst 10 Predicted Labels:", pred_list[:10])
    print("\nFirst 10 Explanations:")
    for i, expl in enumerate(explanations[:10]):
        print(f"{i+1}: {expl}")

if __name__ == "__main__":
    main()
