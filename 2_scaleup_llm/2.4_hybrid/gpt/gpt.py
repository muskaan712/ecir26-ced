import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import openai

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

# GPT-4o config
OPENAI_MODEL = "gpt-4o"
EXPL_TOKENS = 100
TEMPERATURE = 0.1
FEW_SHOT_ERR = 3
FEW_SHOT_NOT = 3
openai.api_key = "sk-proj-eOGkRdhQf1kLt0eOhMAaT3BlbkFJ3cKHnSymxwTdakeiMwze"

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

def load_few_shot_examples(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["id", "src", "mt", "toklabels", "label"])
    err = df[df.label == "ERR"].sample(FEW_SHOT_ERR, random_state=42)
    not_ = df[df.label == "NOT"].sample(FEW_SHOT_NOT, random_state=42)
    err["explanation"] = "Major meaning error or omission detected."
    not_["explanation"] = "Meaning preserved; only minor stylistic issues."
    examples = pd.concat([err.assign(label="ERR")[["src", "mt", "label", "explanation"]], not_.assign(label="NOT")[["src", "mt", "label", "explanation"]]], ignore_index=True)
    return examples.to_dict("records")

SYSTEM_PROMPT = ("You are a professional translation quality analyst. Given an English sentence (EN), its German translation (DE), and a predicted label, "
    "explain in ≤ 50 words why the translation is erroneous (ERR) or acceptable (NOT).")
def build_explanation_messages(src: str, mt: str, label: str, few_shot):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot:
        msgs.append({"role": "user", "content": f"EN: {ex['src']}\nDE: {ex['mt']}\nPREDICTED: {ex['label']}"})
        msgs.append({"role": "assistant", "content": ex["explanation"]})
    msgs.append({"role": "user", "content": f"EN: {src}\nDE: {mt}\nPREDICTED: {label}"})
    return msgs

def generate_explanations(src_list, mt_list, pred_list, few_shot):
    explanations, total_tokens = [], 0
    for src, mt, pred in tqdm(zip(src_list, mt_list, pred_list), total=len(src_list), desc="Generating Explanations"):
        label_str = "ERR" if pred == 1 else "NOT"
        msgs = build_explanation_messages(src, mt, label_str, few_shot)
        resp = openai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, max_tokens=EXPL_TOKENS, temperature=TEMPERATURE)
        explanations.append(resp.choices[0].message.content.strip())
        total_tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
    cost = (total_tokens / 1000) * 0.002  # GPT-4o price per 1K tokens ($)
    print(f"\nTotal GPT tokens: {total_tokens}  →  Estimated cost: ${cost:.4f}\n")
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

    # GPT-4o for explainability
    few_shot = load_few_shot_examples(TRAIN_TSV)
    explanations = generate_explanations(src_list, mt_list, pred_list, few_shot)

    print("\nFirst 10 Predicted Labels:", pred_list[:10])
    print("\nFirst 10 Explanations:")
    for i, expl in enumerate(explanations[:10]):
        print(f"{i+1}: {expl}")

if __name__ == "__main__":
    main()
