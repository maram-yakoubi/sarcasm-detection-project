# -*- coding: utf-8 -*-
"""Sarcasm Detection - MUSTARD Dataset with IFM"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for ifm import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ifm import IntelligentFusionMethods

# ----------------------------
# CONFIGURATION
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ----------------------------
# 1. DATA LOADING AND PREPARATION
# ----------------------------
def load_mustard_dataset(file_path):
    """Load the MUSTARD dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    texts = []
    labels = []

    for key, value in data.items():
        text = value.get('utterance', '').strip()
        label = int(value.get('sarcasm', 0))

        if text and len(text) > 2:
            texts.append(text)
            labels.append(label)

    df = pd.DataFrame({'text': texts, 'label': labels})
    print(f"MUSTARD Dataset: {len(df)} examples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return df

# Load data
mustard_path = '/kaggle/input/mustard-multimodal-sarcasm-detection-dataset/sarcasm_data.json'
df = load_mustard_dataset(mustard_path)

# Train/val/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, stratify=train_df['label'])

print(f"\nData split:")
print(f"   Train: {len(train_df)} examples")
print(f"   Validation: {len(val_df)} examples")
print(f"   Test: {len(test_df)} examples")

# ----------------------------
# 2. DISTILBERT FINE-TUNING (OPTIMIZED)
# ----------------------------
distilbert_model_name = "distilbert-base-uncased"

# Tokenizer configuration
tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer configured - Pad token: {tokenizer.pad_token}")

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Prepare datasets
train_dataset = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
test_dataset = SarcasmDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

# Model initialization
model = AutoModelForSequenceClassification.from_pretrained(
    distilbert_model_name,
    num_labels=2
).to(device)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training configuration
training_args = TrainingArguments(
    output_dir="./distilbert-mustard",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to=None,
    save_total_limit=2,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Training DistilBERT...")
trainer.train()

# Save model
trainer.save_model("./distilbert-mustard")
print("DistilBERT trained and saved!")

# ----------------------------
# 3. MISTRAL LOADING
# ----------------------------
print("Loading Mistral 7B...")
try:
    mistral_pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16,
        max_new_tokens=10,
        do_sample=False,
        temperature=0.0
    )
    print("Mistral 7B loaded successfully!")
except Exception as e:
    print(f"Error loading Mistral: {e}")
    mistral_pipe = None

# ----------------------------
# 4. MUSTARD-SPECIFIC LINGUISTIC RULES
# ----------------------------
def analyze_linguistic_rules_mustard(text):
    """MUSTARD-specific rules based on dataset analysis"""
    text_lower = text.lower()
    score = 0

    # MUSTARD-specific sarcasm patterns
    mustard_patterns = {
        "love how": 2, "so excited": 2, "big surprise": 3, "of course": 3,
        "oh great": 3, "wonderful": 2, "perfect": 2, "fantastic": 2,
        "really": 1, "sure": 2, "obviously": 2, "clearly": 2,
        "what a": 2, "how": 1, "finally": 1, "again": 1, "another": 2,
        "as always": 2, "right on time": 2, "my favorite": 2
    }

    for pattern, points in mustard_patterns.items():
        if pattern in text_lower:
            score += points

    # Structural analysis
    if "!" in text and any(word in text_lower for word in ["great", "wonderful", "perfect", "fantastic", "love"]):
        score += 2

    if "?" in text and any(word in text_lower for word in ["great", "wonderful", "perfect"]):
        score += 1

    word_count = len(text.split())
    threshold = 2 if word_count > 3 else 1.5

    return 1 if score >= threshold else 0, score

# ----------------------------
# 5. EXTENDED FUSION FUNCTIONS (MUSTARD-SPECIFIC)
# ----------------------------

def mustard_parallel_fusion(distil_probs, mistral_pred, mistral_conf, rule_score):
    """Parallel fusion with MUSTARD-specific weighting"""
    if mistral_pred == 1:
        mistral_probs = np.array([1 - mistral_conf, mistral_conf])
    else:
        mistral_probs = np.array([mistral_conf, 1 - mistral_conf])

    # MUSTARD-specific weights
    distil_weight = 1.0
    mistral_weight = mistral_conf * 1.3

    if rule_score >= 3:
        mistral_weight += 0.3

    total_weight = distil_weight + mistral_weight
    fused_probs = (distil_probs * distil_weight + mistral_probs * mistral_weight) / total_weight

    return int(np.argmax(fused_probs)), float(np.max(fused_probs))

def mustard_sequential_fusion(distil_probs, distil_conf, mistral_pred, mistral_conf, text):
    """Sequential fusion with MUSTARD-specific logic"""
    rule_pred, rule_score = analyze_linguistic_rules_mustard(text)

    # 1. Strong rules + Mistral
    if rule_score >= 4 and mistral_pred == 1:
        return 1, max(distil_conf, mistral_conf), "Strong Rules + Mistral"

    # 2. DistilBERT very confident
    if distil_conf > 0.85:
        return int(np.argmax(distil_probs)), float(distil_conf), "DistilBERT (very confident)"

    # 3. Mistral very confident
    if mistral_conf > 0.85:
        return int(mistral_pred), float(mistral_conf), "Mistral (very confident)"

    # 4. Rules + model agreement
    if rule_pred == 1 and (mistral_pred == 1 or np.argmax(distil_probs) == 1):
        confidence_boost = 0.1 if rule_score >= 3 else 0.05
        return 1, float(max(distil_conf, mistral_conf) + confidence_boost), "Rules + Model"

    # 5. Model agreement
    if np.argmax(distil_probs) == mistral_pred:
        return int(mistral_pred), float((distil_conf + mistral_conf) / 2), "Model agreement"

    # 6. Default
    return int(np.argmax(distil_probs)), float(distil_conf * 0.9), "DistilBERT (default)"

def mustard_intelligent_fusion(text):
    """Complete Intelligent Fusion with all methods for MUSTARD"""
    
    # Get DistilBERT prediction
    distil_pred, distil_conf, distil_probs = ifm.get_distilbert_prediction(text)
    
    # Get Mistral prediction
    mistral_pred, mistral_conf = ifm.predict_with_mistral(text)
    
    # Get MUSTARD-specific rules
    rule_pred, rule_score = analyze_linguistic_rules_mustard(text)
    
    # Parallel fusion
    parallel_pred, parallel_conf = mustard_parallel_fusion(distil_probs, mistral_pred, mistral_conf, rule_score)
    
    # Sequential fusion
    sequential_pred, sequential_conf, sequential_method = mustard_sequential_fusion(
        distil_probs, distil_conf, mistral_pred, mistral_conf, text
    )
    
    # Intelligent fusion with all methods
    weights = {
        'distilbert': 0.50,
        'mistral': 0.25,
        'rules': 0.10,
        'sequential': 0.15
    }
    
    # Convert to probabilities
    if mistral_pred == 1:
        mistral_probs = np.array([1 - mistral_conf, mistral_conf])
    else:
        mistral_probs = np.array([mistral_conf, 1 - mistral_conf])
    
    # Rule probabilities
    if rule_score >= 4:
        rule_probs = np.array([0.1, 0.9])
    elif rule_score >= 2:
        rule_probs = np.array([0.3, 0.7])
    elif rule_score >= 1:
        rule_probs = np.array([0.4, 0.6])
    else:
        rule_probs = np.array([0.5, 0.5])
    
    # Sequential probabilities
    if sequential_pred == 1:
        seq_probs = np.array([1 - sequential_conf, sequential_conf])
    else:
        seq_probs = np.array([sequential_conf, 1 - sequential_conf])
    
    # Final weighted fusion
    final_probs = (
        distil_probs * weights['distilbert'] +
        mistral_probs * weights['mistral'] +
        rule_probs * weights['rules'] +
        seq_probs * weights['sequential']
    )
    
    final_pred = int(np.argmax(final_probs))
    final_conf = float(np.max(final_probs))
    
    # Identify dominant method
    method_contributions = {
        'DistilBERT': weights['distilbert'] * distil_probs[final_pred],
        'Mistral': weights['mistral'] * mistral_probs[final_pred],
        'Rules': weights['rules'] * rule_probs[final_pred],
        'Sequential Fusion': weights['sequential'] * seq_probs[final_pred]
    }
    dominant_method = max(method_contributions.items(), key=lambda x: x[1])
    
    return {
        'final_pred': final_pred,
        'final_conf': final_conf,
        'method': f"Fusion - {dominant_method[0]} dominant",
        'distil_pred': distil_pred,
        'distil_conf': distil_conf,
        'mistral_pred': mistral_pred,
        'mistral_conf': mistral_conf,
        'rule_pred': rule_pred,
        'rule_score': rule_score,
        'parallel_pred': parallel_pred,
        'parallel_conf': parallel_conf,
        'sequential_pred': sequential_pred,
        'sequential_conf': sequential_conf,
        'sequential_method': sequential_method,
        'final_probs': final_probs,
        'method_contributions': method_contributions
    }

# ----------------------------
# 6. INITIALIZE IFM
# ----------------------------
ifm = IntelligentFusionMethods(model, tokenizer, mistral_pipe)

# ----------------------------
# 7. FULL EVALUATION ON TEST SET
# ----------------------------
print(f"\n===== EVALUATION ON {len(test_df)} MUSTARD EXAMPLES =====")

y_true = test_df['label'].tolist()

# Initialize prediction containers - ALL METHODS
predictions = {
    "DistilBERT": [],
    "Mistral": [],
    "Linguistic Rules": [],
    "Parallel Fusion": [],
    "Sequential Fusion": [],
    "Complete Intelligent Fusion": []
}

# Compute predictions
success_count = 0
for i, (text, true_label) in enumerate(tqdm(zip(test_df['text'].tolist(), y_true), desc="Evaluation", total=len(test_df))):
    try:
        # Get predictions using IFM
        distil_pred, distil_conf, distil_probs = ifm.get_distilbert_prediction(text)
        mistral_pred, mistral_conf = ifm.predict_with_mistral(text)
        rule_pred, rule_score = analyze_linguistic_rules_mustard(text)
        
        # Parallel fusion
        parallel_pred, parallel_conf = mustard_parallel_fusion(distil_probs, mistral_pred, mistral_conf, rule_score)
        
        # Sequential fusion
        sequential_pred, sequential_conf, sequential_method = mustard_sequential_fusion(
            distil_probs, distil_conf, mistral_pred, mistral_conf, text
        )
        
        # Complete intelligent fusion
        fusion_result = mustard_intelligent_fusion(text)
        
        # Store results
        predictions["DistilBERT"].append(distil_pred)
        predictions["Mistral"].append(mistral_pred)
        predictions["Linguistic Rules"].append(rule_pred)
        predictions["Parallel Fusion"].append(parallel_pred)
        predictions["Sequential Fusion"].append(sequential_pred)
        predictions["Complete Intelligent Fusion"].append(fusion_result['final_pred'])
        
        success_count += 1

    except Exception as e:
        print(f"Error on sample {i}: {e}")
        for key in predictions:
            predictions[key].append(0)

print(f"{success_count}/{len(test_df)} predictions successful")

# ----------------------------
# 8. DETAILED METRICS COMPUTATION
# ----------------------------
def compute_comprehensive_metrics(y_true, y_pred, method_name):
    """Compute full metrics for a given method"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print(f"\n===== DETAILED PERFORMANCE - ALL METHODS =====")
print(f"{'Method':40s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s}")
print("-" * 80)

results = {}
for method_name, y_pred in predictions.items():
    metrics = compute_comprehensive_metrics(y_true, y_pred, method_name)
    results[method_name] = metrics
    print(f"{method_name:40s} {metrics['accuracy']:8.3f} {metrics['precision']:10.3f} {metrics['recall']:8.3f} {metrics['f1']:6.3f}")

# ----------------------------
# 9. FIND BEST METHOD
# ----------------------------
best_method = max(results.items(), key=lambda x: x[1]['f1'])

print(f"\n===== FINAL REPORT - MUSTARD DATASET =====")
print(f"BEST METHOD: {best_method[0]}")
print(f"   F1-score:  {best_method[1]['f1']:.3f}")
print(f"   Accuracy:  {best_method[1]['accuracy']:.3f}")
print(f"   Precision: {best_method[1]['precision']:.3f}")
print(f"   Recall:    {best_method[1]['recall']:.3f}")

print(f"\nMETHOD RANKING (sorted by F1-score):")
for method_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
    print(f"   {method_name:40s} F1: {metrics['f1']:.3f} | Acc: {metrics['accuracy']:.3f}")

print(f"\n✅ MUSTARD EVALUATION COMPLETE!")