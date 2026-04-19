# -*- coding: utf-8 -*-
"""IRONIE DÉTECTION - 3 FUSIONS - ÉVALUATION 100% TEST (212 ÉCHANT.)"""
import os
import torch
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# CONFIGURATION
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ----------------------------
# 1. CHARGEMENT + RÉÉQUILIBRAGE 50-50 (VOTRE DATASET)
# ----------------------------
def load_and_balance_irony_data(file_path):
    print(f"Chargement du dataset depuis: {file_path}")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['comment_text', 'label'])
    df['comment_text'] = df['comment_text'].astype(str).str.strip()
    df = df[df['comment_text'] != ""]
    df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
    
    print(f"Distribution originale: {df['label'].value_counts().to_dict()}")
    
    # Rééquilibrage 50-50
    df_ironic = df[df['label'] == 1]
    df_normal = df[df['label'] == 0]
    df_ironic_balanced = resample(df_ironic, replace=True, n_samples=len(df_normal), random_state=42)
    df_balanced = pd.concat([df_normal, df_ironic_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset final (équilibré): {len(df_balanced):,} échantillons")
    print(f"Distribution: {df_balanced['label'].value_counts().to_dict()}")
    return df_balanced

full_df = load_and_balance_irony_data('/kaggle/input/ironic-corpus/irony-labeled.csv')

train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42, stratify=full_df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"\nRÉPARTITION:")
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# ----------------------------
# 2. DISTILBERT FINE-TUNING (ADAPTÉ PETIT DATASET)
# ----------------------------
distilbert_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    distilbert_model_name, num_labels=2
).to(device)

class IronyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts; self.labels = labels; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_data = IronyDataset(train_df['comment_text'].tolist(), train_df['label'].tolist(), tokenizer)
val_data = IronyDataset(val_df['comment_text'].tolist(), val_df['label'].tolist(), tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

training_args = TrainingArguments(
    output_dir="./distilbert-irony-small",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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

print("\nENTRAÎNEMENT DISTILBERT (petit dataset)...")
trainer.train()

# ----------------------------
# 3. MISTRAL FEW-SHOT (CORRIGÉ)
# ----------------------------
try:
    mistral_pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16,
        max_new_tokens=15,
        temperature=0.1,
        do_sample=False,
    )
    print("Mistral chargé")
except:
    mistral_pipe = None
    print("Mistral non disponible")

def predict_with_mistral(text):
    if mistral_pipe is None:
        return 0, 0.5
    prompt = f"""<s>[INST] Analyse ce commentaire. Réponds UNIQUEMENT par "IRONIQUE" ou "NORMAL".
EXEMPLES:
"thanks for sharing" → NORMAL
"oh great, another meeting" → IRONIQUE
Commentaire: "{text[:140]}"
Réponse: [/INST]"""
    try:
        out = mistral_pipe(prompt)
        resp = out[0]['generated_text'].split("Réponse:")[-1].strip().upper()
        if "IRONIQUE" in resp:
            return 1, 0.92
        else:
            return 0, 0.92
    except:
        return 0, 0.5

# ----------------------------
# 4. RÈGLES LINGUISTIQUES
# ----------------------------
def detect_irony_rules(text):
    t = text.lower()
    patterns = [
        r"\boh\s+great\b", r"\bof\s+course\b", r"\bsure[, ]", r"\bobviously\b",
        r"\bcool story", r"\byeah right\b", r"\bjust what", r"\bi'?m\s+shocked\b"
    ]
    return 1 if any(re.search(p, t) for p in patterns) else 0, 0.85 if any(re.search(p, t) for p in patterns) else 0.1

# ----------------------------
# 5. DISTILBERT PREDICTION
# ----------------------------
def get_distilbert_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf, probs

# ----------------------------
# 6. TROIS FUSIONS
# ----------------------------
def fusion_parallele(distil_probs, mistral_pred, mistral_conf):
    mistral_probs = np.array([1 - mistral_conf, mistral_conf]) if mistral_pred == 1 else np.array([mistral_conf, 1 - mistral_conf])
    fused = (distil_probs + mistral_probs) / 2
    return int(np.argmax(fused)), float(np.max(fused))

def fusion_sequentielle(distil_pred, distil_conf, mistral_pred, mistral_conf, rules_pred, rules_conf):
    if distil_conf > 0.90:
        return distil_pred, distil_conf, "DistilBERT"
    if mistral_conf > 0.90 and rules_pred == 1:
        return 1, mistral_conf, "Mistral+Règles"
    if distil_pred == mistral_pred and distil_conf > 0.70:
        return distil_pred, (distil_conf + mistral_conf)/2, "Accord"
    return distil_pred, distil_conf, "DistilBERT"

def fusion_intelligente(text):
    d_pred, d_conf, d_probs = get_distilbert_prediction(text)
    m_pred, m_conf = predict_with_mistral(text)
    r_pred, r_conf = detect_irony_rules(text)
    
    p_pred, p_conf = fusion_parallele(d_probs, m_pred, m_conf)
    s_pred, s_conf, _ = fusion_sequentielle(d_pred, d_conf, m_pred, m_conf, r_pred, r_conf)
   
    if d_conf > 0.85:
        return d_pred, d_conf, "DistilBERT"
    elif p_conf > s_conf + 0.05:
        return p_pred, p_conf, "Fusion Parallèle"
    elif s_conf > p_conf + 0.05:
        return s_pred, s_conf, "Fusion Séquentielle"
    else:
        return d_pred, d_conf * 0.9, "DistilBERT (défaut)"

# ----------------------------
# 7. ÉVALUATION 100% TEST (212 ÉCHANT.)
# ----------------------------
print(f"\nÉVALUATION SUR 100% DU TEST: {len(test_df):,} échantillons")
y_true = test_df['label'].tolist()

y_pred_distil = []
y_pred_mistral = []
y_pred_parallele = []
y_pred_sequentielle = []
y_pred_fusion = []

for text in tqdm(test_df['comment_text'].tolist(), desc="Évaluation"):
    d_pred, d_conf, d_probs = get_distilbert_prediction(text)
    m_pred, m_conf = predict_with_mistral(text)
    r_pred, r_conf = detect_irony_rules(text)
    
    p_pred, p_conf = fusion_parallele(d_probs, m_pred, m_conf)
    s_pred, s_conf, _ = fusion_sequentielle(d_pred, d_conf, m_pred, m_conf, r_pred, r_conf)
    f_pred, f_conf, _ = fusion_intelligente(text)
   
    y_pred_distil.append(d_pred)
    y_pred_mistral.append(m_pred)
    y_pred_parallele.append(p_pred)
    y_pred_sequentielle.append(s_pred)
    y_pred_fusion.append(f_pred)

# ----------------------------
# 8. MÉTRIQUES
# ----------------------------
def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return round(acc, 4), round(p, 4), round(r, 4), round(f1, 4)

methods = {
    "DistilBERT": y_pred_distil,
    "Mistral Few-Shot": y_pred_mistral,
    "Fusion Parallèle": y_pred_parallele,
    "Fusion Séquentielle": y_pred_sequentielle,
    "Fusion Intelligente": y_pred_fusion
}

print(f"\n===== RÉSULTATS - 100% TEST ({len(test_df):,} échantillons) =====")
print(f"{'Méthode':28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-" * 56)
for name, pred in methods.items():
    acc, p, r, f1 = metrics(y_true, pred)
    print(f"{name:28} {acc:6.4f} {p:6.4f} {r:6.4f} {f1:6.4f}")

# ----------------------------
# 9. RÉSUMÉ
# ----------------------------
print(f"\nRÉSUMÉ:")
print(f"Dataset: irony-labeled.csv → {len(full_df):,} échantillons équilibrés")
print(f"Test: 100% → {len(test_df):,} échantillons")
print(f"3 FUSIONS COMPLÈTES + ÉVALUATION TOTALE")
print(f"CODE PRÊT, ROBUSTE, ADAPTÉ À VOTRE DATASET")





===== RÉSULTATS - 100% TEST (424 échantillons) =====
Méthode                         Acc   Prec    Rec     F1
--------------------------------------------------------
DistilBERT                   0.8066 0.8066 0.8066 0.8066
Mistral Few-Shot             0.5354 0.5200 0.9198 0.6644
Fusion Parallèle             0.7335 0.6854 0.8632 0.7641
Fusion Séquentielle          0.8042 0.8000 0.8113 0.8056
Fusion Intelligente          0.8113 0.8056 0.8208 0.8131

RÉSUMÉ:
Dataset: irony-labeled.csv → 2,824 échantillons équilibrés
Test: 100% → 424 échantillons
3 FUSIONS COMPLÈTES + ÉVALUATION TOTALE


https://www.kaggle.com/code/maramyakoubi/notebookc6fb9878e5/edit