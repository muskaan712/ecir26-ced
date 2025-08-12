import os
import json
import re
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# ---- Limited Grid Search Configuration (3 models, <1 hour) ----
data_dir = "/home/s13mchop/LLMs/data/wmt21"
model_configs = {
    "BERT": {
        "model_name": "google/bert_uncased_L-12_H-768_A-12",
        "learning_rates": [2e-5, 3e-5],  
        "batch_size": 32,               # Single batch size for A100
        "epochs": 3                     # Fixed epochs
    },
    "ModernBERT-base": {
        "model_name": "answerdotai/ModernBERT-base",
        "learning_rates": [2e-5, 3e-5],  
        "batch_size": 32,               # Single batch size
        "epochs": 3                     # Fixed epochs
    },
    "ModernBERT-large": {
        "model_name": "answerdotai/ModernBERT-large",
        "learning_rates": [2e-5, 3e-5],  
        "batch_size": 16,               # Smaller for large model
        "epochs": 3                     # Fixed epochs
    }
}
output_dir = "output_three_models"
max_length = 256
device = torch.device("cuda")
warmup_ratio = 0.1
optimizer_betas = (0.9, 0.999)
optimizer_eps = 1e-6
random_seeds = [42, 123]  # 2 seeds for minimal variance

# ---- Text cleaning function ----
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

# ---- Dataset (TSV without headers) ----
class CEDDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_length=256):
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                         names=['id','src','mt','toklabels','label'], dtype={0:int,1:str,2:str,4:str})
        self.src = df['src'].astype(str).tolist()
        self.mt = df['mt'].astype(str).tolist()
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
    return {'mcc': mcc, 'f1_err': f1[0], 'f1_not': f1[1], 'precision_err': precision[0], 'recall_err': recall[0]}

# ---- Training Function (No Early Stopping) ----
def train_model(model_name, model_path, lr, batch_size, epochs, seed):
    print(f"\nTraining {model_name} - LR: {lr}, Batch: {batch_size}, Seed: {seed}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup data
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_file = os.path.join(data_dir, 'ende_majority_train.tsv')
    dev_file = os.path.join(data_dir, 'ende_majority_dev.tsv')
    
    train_ds = CEDDataset(train_file, tokenizer, max_length)
    dev_ds = CEDDataset(dev_file, tokenizer, max_length)
    
    # Data loaders
    labels = train_ds.labels
    class_counts = [labels.count(0), labels.count(1)]
    total = sum(class_counts)
    weights = [total/(2*c) for c in class_counts]
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    
    # Model setup
    model = CEDModel(model_path).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=optimizer_betas, eps=optimizer_eps)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    best_dev_mcc = -1
    final_metrics = None
    
    # Training loop (complete all epochs, no early stopping)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch}", leave=False):
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
        
        # Evaluate and track best
        dev_metrics = evaluate(model, dev_loader, device)
        if dev_metrics['mcc'] > best_dev_mcc:
            best_dev_mcc = dev_metrics['mcc']
            final_metrics = dev_metrics
        
        print(f"  Epoch {epoch}: Loss={epoch_loss/len(train_loader):.4f}, Dev MCC={dev_metrics['mcc']:.4f}")
    
    return final_metrics

# ---- Grid Search for Single Model ----
def grid_search_model(model_name, config):
    print(f"\n{'='*50}")
    print(f"Grid Search for {model_name}")
    print(f"{'='*50}")
    
    results = []
    
    for lr in config["learning_rates"]:
        seed_results = []
        
        for seed in random_seeds:
            try:
                metrics = train_model(
                    model_name,
                    config["model_name"], 
                    lr,
                    config["batch_size"],
                    config["epochs"],
                    seed
                )
                seed_results.append(metrics)
            except Exception as e:
                print(f"Error with {model_name}, lr={lr}, seed={seed}: {e}")
                continue
        
        if seed_results:
            # Average across seeds
            avg_metrics = {
                'mcc': np.mean([m['mcc'] for m in seed_results]),
                'mcc_std': np.std([m['mcc'] for m in seed_results]),
                'f1_err': np.mean([m['f1_err'] for m in seed_results]),
                'f1_err_std': np.std([m['f1_err'] for m in seed_results]),
                'f1_not': np.mean([m['f1_not'] for m in seed_results]),
                'f1_not_std': np.std([m['f1_not'] for m in seed_results])
            }
            result = {
                'model': model_name,
                'lr': lr,
                'batch_size': config["batch_size"],
                'epochs': config["epochs"],
                'metrics': avg_metrics,
                'num_seeds': len(seed_results)
            }
            results.append(result)
            print(f"LR {lr}: MCC={avg_metrics['mcc']:.4f}±{avg_metrics['mcc_std']:.4f}, "
                  f"F1-ERR={avg_metrics['f1_err']:.4f}±{avg_metrics['f1_err_std']:.4f}")
    
    return results

# ---- Main Function ----
def main():
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    
    print("=== THREE MODEL COMPARISON (Limited Grid Search) ===")
    total_runs = sum(len(config["learning_rates"]) * len(random_seeds) for config in model_configs.values())
    print(f"Total training runs: {total_runs}")
    print(f"Expected epochs: ~{total_runs * 3}")
    print(f"Estimated time: ~{total_runs * 3 * 1.5:.0f} minutes")
    
    # Run grid search for each model
    for model_name, config in model_configs.items():
        model_results = grid_search_model(model_name, config)
        all_results.extend(model_results)
    
    end_time.record()
    torch.cuda.synchronize()
    total_time = start_time.elapsed_time(end_time) / (1000 * 60)  # Convert to minutes
    
    # Find best configuration for each model
    model_best = {}
    for model_name in model_configs.keys():
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            best_result = max(model_results, key=lambda x: x['metrics']['mcc'])
            model_best[model_name] = best_result
    
    # Save all results
    with open(os.path.join(output_dir, 'three_models_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print final comparison
    print(f"\n{'='*70}")
    print("FINAL THREE-MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Total Runtime: {total_time:.1f} minutes")
    
    for model_name, best_result in model_best.items():
        metrics = best_result['metrics']
        print(f"\n{model_name} (Best Config: LR={best_result['lr']}):")
        print(f"  MCC: {metrics['mcc']:.4f} ± {metrics['mcc_std']:.4f}")
        print(f"  F1-ERR: {metrics['f1_err']:.4f} ± {metrics['f1_err_std']:.4f}") 
        print(f"  F1-NOT: {metrics['f1_not']:.4f} ± {metrics['f1_not_std']:.4f}")
    
    # Rank models
    ranked_models = sorted(model_best.items(), key=lambda x: x[1]['metrics']['mcc'], reverse=True)
    print(f"\n{'='*70}")
    print("FINAL RANKING (by MCC):")
    print("="*70)
    for i, (model_name, result) in enumerate(ranked_models, 1):
        mcc = result['metrics']['mcc']
        mcc_std = result['metrics']['mcc_std']
        lr = result['lr']
        print(f"{i}. {model_name}: {mcc:.4f} ± {mcc_std:.4f} (LR: {lr})")
    
    print(f"\n🏆 WINNER: {ranked_models[0][0]}")
    print(f"Results saved to {output_dir}/three_models_results.json")

if __name__ == "__main__":
    main()
