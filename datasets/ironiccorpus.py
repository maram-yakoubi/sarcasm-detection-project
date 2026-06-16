# -*- coding: utf-8 -*-
"""Irony Detection - Irony Corpus Dataset with IFM"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, pipeline
)
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for ifm import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ifm import IntelligentFusionMethods

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ----------------------------
# 1. DATA LOADING
# ----------------------------
print("\n" + "="*80)
print("IRONY CORPUS DATASET")
print("="*80)

# Load dataset - adjust path as needed
df = pd.read_csv('/kaggle/input/ironic-corpus/irony-labeled.csv')
df = df.dropna(subset=['comment_text', 'label'])
df['comment_text'] = df['comment_text'].astype(str).str.strip()
df = df[df['comment_text'] != ""]
df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
df = df.rename(columns={'comment_text': 'text'})

print(f"Dataset size: {len(df):,}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Split data
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.176, random_state=42, stratify=train_df['label'])

print(f"\nSplit: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

# ----------------------------
# 2. DISTILBERT FINE-TUNING
# ----------------------------
print("\n" + "="*60)
print("DISTILBERT FINE-TUNING")
print("="*60)

class IronyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Prepare datasets
train_data = IronyDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_data = IronyDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./distilbert-irony",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_steps=50,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

print("Starting DistilBERT training...")
trainer.train()

# ----------------------------
# 3. MISTRAL LOADING
# ----------------------------
print("\n" + "="*60)
print("MISTRAL-7B LOADING")
print("="*60)

try:
    mistral_pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16,
        max_new_tokens=10,
        do_sample=False
    )
    print("Mistral loaded successfully")
except Exception as e:
    print(f"Unable to load Mistral: {e}")
    mistral_pipe = None

# ----------------------------
# 4. EVALUATION
# ----------------------------
print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60)

# Initialize IFM
ifm = IntelligentFusionMethods(model, tokenizer, mistral_pipe)

# Prepare results
y_true = test_df['label'].tolist()
results = {
    'distilbert': [],
    'mistral': [],
    'parallel': [],
    'sequential': [],
    'intelligent': []
}

print(f"Testing on {len(test_df):,} samples...")
pbar = tqdm(total=len(test_df), desc="Evaluating")

for _, row in test_df.iterrows():
    text = row['text']
    
    # Get DistilBERT prediction
    distil_pred, distil_conf, distil_probs = ifm.get_distilbert_prediction(text)
    
    # Get Mistral prediction
    mistral_pred, mistral_conf = ifm.predict_with_mistral(text)
    
    # Parallel fusion
    parallel_pred, parallel_conf = ifm.parallel_fusion(distil_probs, mistral_pred, mistral_conf)
    
    # Sequential fusion
    sequential_pred, sequential_conf, seq_method = ifm.sequential_fusion(
        distil_probs, distil_conf, mistral_pred, mistral_conf, text
    )
    
    # Intelligent fusion
    result = ifm.intelligent_fusion(text)
    
    # Store results
    results['distilbert'].append(distil_pred)
    results['mistral'].append(mistral_pred)
    results['parallel'].append(parallel_pred)
    results['sequential'].append(sequential_pred)
    results['intelligent'].append(result['final_pred'])
    
    pbar.update(1)

pbar.close()

# ----------------------------
# 5. RESULTS
# ----------------------------
print("\n" + "="*80)
print("FINAL RESULTS - IRONY CORPUS")
print("="*80)

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)

methods = {
    "DistilBERT": results['distilbert'],
    "Mistral-7B (Few-Shot)": results['mistral'],
    "Parallel Fusion": results['parallel'],
    "Sequential Fusion": results['sequential'],
    "Intelligent Fusion": results['intelligent']
}

print(f"\n{'Method':30s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
print("-" * 65)

for name, preds in methods.items():
    acc, prec, rec, f1 = calculate_metrics(y_true, preds)
    print(f"{name:30s} {acc:8.4f} {prec:10.4f} {rec:8.4f} {f1:8.4f}")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)