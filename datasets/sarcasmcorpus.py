# -*- coding: utf-8 -*-
"""Sarcasm Detection - Oraby et al. Datasets (RQ/GEN/HYP) with IFM"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, TrainerCallback,
    pipeline
)

# Add parent directory to path for ifm import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ifm import IntelligentFusionMethods

# ----------------------------
# OPTIMIZED CONFIG FOR GPU
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ----------------------------
# 1) DATASET LOADING
# ----------------------------
def load_orabi_dataset(file_path):
    file_path = Path(file_path)
    print(f"\nLoading {file_path.name} ...")
    df = pd.read_csv(file_path)
    print(f"   CSV loaded: {len(df)} rows")
    if 'class' not in df.columns:
        raise ValueError("The CSV must contain a 'class' column")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df['label'] = df['class'].map({'notsarc': 0, 'sarc': 1})
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""].reset_index(drop=True)
    print(f"   After cleaning: {len(df)} samples")
    print(f"   Distribution: {df['class'].value_counts().to_dict()}")
    return df[['text', 'label', 'class']]

# Dataset paths
dataset_paths = {
    "RQ": "/kaggle/input/sarcasm-corpus-v2oraby-et-al/RQ-sarc-notsarc.csv",
    "GEN": "/kaggle/input/sarcasm-corpus-v2oraby-et-al/GEN-sarc-notsarc.csv", 
    "HYP": "/kaggle/input/sarcasm-corpus-v2oraby-et-al/HYP-sarc-notsarc.csv"
}

print("STARTING DATASET LOADING...")
datasets = {}
for name, path in dataset_paths.items():
    if os.path.exists(path):
        datasets[name] = load_orabi_dataset(path)
    else:
        print(f"File not found: {path} (skipped)")

print(f"\nLOADING COMPLETE: {len(datasets)} datasets available")

# ----------------------------
# 2) OPTIMIZED PyTorch Dataset
# ----------------------------
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, dataset_name=""):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ----------------------------
# 3) Callback with memory monitoring
# ----------------------------
class MemoryMonitoringCallback(TrainerCallback):
    def __init__(self, dataset_name=""):
        self.dataset_name = dataset_name

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step % 50 == 0:
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"   Step {state.global_step} - GPU: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

# ----------------------------
# 4) Metrics
# ----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    if len(np.unique(labels)) == 1:
        prec = recall = f1 = 0.0
    else:
        prec, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1}

# ----------------------------
# 5) OPTIMIZED DistilBERT Training
# ----------------------------
def train_distilbert_on_dataset(train_df, val_df, dataset_name, model_name="distilbert-base-uncased"):
    print("\n" + "="*60)
    print(f"DISTILBERT TRAINING - {dataset_name}")
    print("="*60)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("   Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    
    per_device_batch_size = 8
    
    print("   Model & tokenizer loaded")

    train_ds = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, dataset_name=f"Train-{dataset_name}")
    val_ds = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, dataset_name=f"Val-{dataset_name}")

    output_dir = f"./distilbert-{dataset_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        disable_tqdm=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[MemoryMonitoringCallback(dataset_name)]
    )

    print(f"   Starting training...")
    
    try:
        train_result = trainer.train()
        print("   Training complete.")
        
        eval_results = trainer.evaluate()
        print(f"   Final results - F1: {eval_results.get('eval_f1', 0):.4f}")
        
        return model, tokenizer
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("   ERROR: Insufficient GPU memory!")
            return None, None
        else:
            raise e

# ----------------------------
# 6) MISTRAL MANAGER
# ----------------------------
class MistralManager:
    def __init__(self):
        self.pipe = None
        self.is_loaded = False
    
    def initialize(self):
        if self.is_loaded:
            return self.pipe
            
        print("\nInitializing Mistral...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.pipe = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.3",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0
            )
            self.is_loaded = True
            print("   Mistral loaded successfully")
            return self.pipe
        except Exception as e:
            print(f"   Failed to load Mistral: {e}")
            return None
    
    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("   Mistral unloaded from GPU memory")

# ----------------------------
# 7) EVALUATION PIPELINE WITH IFM
# ----------------------------
results_summary = {}

for dataset_name, full_df in datasets.items():
    print("\n" + "="*80)
    print(f"PIPELINE - {dataset_name}")
    print("="*80)

    # Data split
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # DistilBERT training
    distil_model, distil_tokenizer = train_distilbert_on_dataset(train_df, val_df, dataset_name)
    
    if distil_model is None:
        print("   Skipping to next dataset...")
        continue

    # COMPLETE evaluation with all methods
    y_true = test_df['label'].tolist()
    methods_preds = {
        "DistilBERT": [],
        "Mistral Few-Shot": [],
        "Parallel Fusion": [],
        "Sequential Fusion": [], 
        "Intelligent Fusion": []
    }

    print(f"\nCOMPLETE evaluation on {len(test_df)} samples...")
    
    # Load Mistral for evaluation
    mistral_manager = MistralManager()
    mistral_pipe = mistral_manager.initialize()
    
    if mistral_pipe is None:
        print("   Evaluation without Mistral (using fallbacks)")
    
    # Initialize IFM
    ifm = IntelligentFusionMethods(distil_model, distil_tokenizer, mistral_pipe)
    
    pbar = tqdm(total=len(test_df), desc=f"Evaluation {dataset_name}")
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        text = row['text']
        true_label = int(row['label'])

        # Get predictions using IFM
        distil_pred, distil_conf, distil_probs = ifm.get_distilbert_prediction(text)
        mistral_pred, mistral_conf = ifm.predict_with_mistral(text)
        
        # Parallel fusion
        parallel_pred, parallel_conf = ifm.parallel_fusion(distil_probs, mistral_pred, mistral_conf)
        
        # Sequential fusion
        sequential_pred, sequential_conf, sequential_method = ifm.sequential_fusion(
            distil_probs, distil_conf, mistral_pred, mistral_conf, text
        )
        
        # Intelligent fusion
        fusion_result = ifm.intelligent_fusion(text)

        # Store ALL predictions
        methods_preds["DistilBERT"].append(distil_pred)
        methods_preds["Mistral Few-Shot"].append(mistral_pred)
        methods_preds["Parallel Fusion"].append(parallel_pred)
        methods_preds["Sequential Fusion"].append(sequential_pred)
        methods_preds["Intelligent Fusion"].append(fusion_result['final_pred'])

        # Update progress
        if (i + 1) % 50 == 0:
            postfix = {}
            for name in methods_preds.keys():
                if len(methods_preds[name]) > 0:
                    acc = accuracy_score(y_true[:i+1], methods_preds[name])
                    postfix[name[:8]] = f'{acc:.3f}'
            pbar.set_postfix(postfix)
        
        pbar.update(1)
    
    pbar.close()
    
    # Free memory
    mistral_manager.unload()
    del distil_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate final metrics
    print(f"\nFINAL RESULTS ({dataset_name}):")
    print(f"{'Method':25s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s}")
    print("-" * 60)
    
    dataset_results = {}
    for name, preds in methods_preds.items():
        acc = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        print(f"{name:25s} {acc:8.3f} {prec:10.3f} {rec:8.3f} {f1:6.3f}")
        dataset_results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    results_summary[dataset_name] = dataset_results

# ----------------------------
# 8) COMPLETE FINAL REPORT
# ----------------------------
print("\n" + "="*80)
print("FINAL REPORT - ALL FUSION METHODS")
print("="*80)

if results_summary:
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    methods_list = ["DistilBERT", "Mistral Few-Shot", "Parallel Fusion", "Sequential Fusion", "Intelligent Fusion"]

    for metric in metrics:
        print(f"\nCOMPARISON - {metric.upper()}:")
        header = f"{'Dataset':10s}" + "".join([f"{m:>18s}" for m in methods_list])
        print(header)
        print("-" * (10 + 18 * len(methods_list)))
        for ds in results_summary.keys():
            row = f"{ds:10s}"
            for method in methods_list:
                val = results_summary[ds].get(method, {}).get(metric, 0)
                row += f"{val:18.3f}"
            print(row)

    # Find the best global method
    print(f"\nBEST GLOBAL METHOD:")
    best_methods = {}
    for ds in results_summary.keys():
        best_f1 = 0
        best_method = ""
        for method in methods_list:
            f1_score = results_summary[ds].get(method, {}).get('f1', 0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_method = method
        best_methods[ds] = (best_method, best_f1)
        print(f"   {ds}: {best_method} (F1={best_f1:.3f})")

else:
    print("No results to display")

print("\nCOMPLETE PIPELINE FINISHED!")