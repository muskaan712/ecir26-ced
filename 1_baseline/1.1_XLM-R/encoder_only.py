import os
import json
import re
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# ---- Configuration (from KDML-LWDA_2022 settings) ----
data_dir = "/home/s13mchop/LLMs/data/wmt21"  # contains `ende_majority_train.tsv` and `ende_majority_dev.tsv`
model_name = "xlm-roberta-large"
output_dir = "output"
epochs = 3
batch_size = 16              # as in paper
learning_rate = 5e-6         # max LR reached after warmup
max_length = 256
device = torch.device("cuda")  # force GPU
warmup_ratio = 0.1           # 10% of total steps
optimizer_betas = (0.9, 0.999)
optimizer_eps = 1e-6

# ---- Text cleaning function ----
def clean_text(text):
    # remove non-alphanumeric, lowercase
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

# ---- Dataset (TSV without headers) ----
class CEDDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_length=256):
        # TSV columns: id, src, mt, token_labels, seq_label
        df = pd.read_csv(tsv_path, sep='	', header=None,
                         names=['id','src','mt','toklabels','label'], dtype={0:int,1:str,2:str,4:str})
        # Keep only relevant fields
        self.src = df['src'].astype(str).tolist()
        self.mt = df['mt'].astype(str).tolist()
        # Convert textual label (ERR/NOT) to int: ERR->1, NOT->0
        self.labels = df['label'].map({'ERR':1, 'NOT':0}).tolist()
        assert len(self.src) == len(self.mt) == len(self.labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        src_text = clean_text(self.src[idx])
        mt_text = clean_text(self.mt[idx])
        inputs = self.tokenizer(src_text, mt_text,
                                truncation=True,
                                padding="max_length",
                                max_length=self.max_length,
                                return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['src'] = src_text
        item['mt'] = mt_text
        return item

# ---- Model Definition ----
class CEDModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),  # paper uses d=1024 to d=2
            nn.Tanh(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h0 = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(h0)
        return logits

# ---- Evaluation ----
def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].long().to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(logits, dim=-1)
            preds += pred.cpu().tolist()
            trues += labels.cpu().tolist()
    mcc = matthews_corrcoef(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, labels=[1,0], zero_division=0)
    cm = confusion_matrix(trues, preds, labels=[1,0])
    return {'mcc': mcc, 'f1_err': f1[0], 'f1_not': f1[1], 'precision_err': precision[0], 'recall_err': recall[0], 'confusion_matrix': cm.tolist()}

# ---- Main Training Routine ----
def main():
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_file = os.path.join(data_dir, 'ende_majority_train.tsv')
    dev_file = os.path.join(data_dir, 'ende_majority_dev.tsv')
    train_ds = CEDDataset(train_file, tokenizer, max_length)
    dev_ds = CEDDataset(dev_file, tokenizer, max_length)
    test_ds = dev_ds  # use dev as test if no explicit test set

    # Weighted sampling for imbalance
    labels = train_ds.labels
    class_counts = [labels.count(0), labels.count(1)]
    total = sum(class_counts)
    weights = [total/(2*c) for c in class_counts]  # inverse freq
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model, optimizer, scheduler, loss
    model = CEDModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate,
                      betas=optimizer_betas, eps=optimizer_eps)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].long().to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} - avg loss: {epoch_loss/len(train_loader):.4f}")

        dev_metrics = evaluate(model, dev_loader, device)
        print(f"Dev MCC: {dev_metrics['mcc']:.4f}, F1-ERR: {dev_metrics['f1_err']:.4f}, F1-NOT: {dev_metrics['f1_not']:.4f}")

    # Final evaluation
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test MCC: {test_metrics['mcc']:.4f}, F1-ERR: {test_metrics['f1_err']:.4f}, F1-NOT: {test_metrics['f1_not']:.4f}")

    # Save artifacts
    model.save_pretrained(os.path.join(output_dir, 'xlmr-ced-model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'xlmr-ced-model'))
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({'dev': dev_metrics, 'test': test_metrics}, f, indent=2)

    # Save predictions
    with open(os.path.join(output_dir, 'predictions.tsv'), 'w', encoding='utf-8') as f:
        f.write("src	mt	true	pred")
        for batch in tqdm(test_loader, desc="Saving preds"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].tolist()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).tolist()
            for src, mt, t, p in zip(batch['src'], batch['mt'], labels, preds):
                f.write(f"{src}	{mt}	{t}	{p}")

if __name__ == "__main__":
    main()
