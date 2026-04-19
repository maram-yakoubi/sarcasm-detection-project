
import os
import torch
import pandas as pd
import numpy as np
import re
import time
import sys
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from IPython.display import clear_output

# ----------------------------
# CONFIGURATION
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 10000  # ÉVALUATION SUR 10K
BATCH_SIZE_INFERENCE = 64

print(f"Device utilisé : {device}")
sys.stdout.flush()

# ----------------------------
# 1. CHARGEMENT DATASET
# ----------------------------
def load_reddit_sarcasm_data(file_path, sample_frac=0.5):
    print(f"Chargement du dataset depuis : {file_path}")
    df = pd.read_csv(file_path)
    df = df.sample(frac=sample_frac, random_state=42)
    df = df.rename(columns={"comment": "text"})[['text', 'label']]
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""].dropna(subset=['text']).reset_index(drop=True)
    print(f"Dataset chargé : {len(df):,} échantillons (50% du total)")
    print(f"Distribution des classes : {df['label'].value_counts().to_dict()}")
    return df

full_df = load_reddit_sarcasm_data('/kaggle/input/sarcasm/train-balanced-sarcasm.csv', sample_frac=0.5)
train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42, stratify=full_df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"\nRÉPARTITION (50% du dataset original) :")
print(f"Train : {len(train_df):,} | Val : {len(val_df):,} | Test : {len(test_df):,}")
print("\n" + "═" * 75)
sys.stdout.flush()
time.sleep(1)
clear_output(wait=True)

# ----------------------------
# 2. DISTILBERT FINE-TUNING
# ----------------------------
distilbert_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_name, num_labels=2).to(device)

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length',    max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_data = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_data = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

training_args = TrainingArguments(
    output_dir="./distilbert-sarcasm-final",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to=[],
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, compute_metrics=compute_metrics)
print("\nENTRAÎNEMENT DISTILBERT...")
sys.stdout.flush()
trainer.train()
print("ENTRAÎNEMENT TERMINÉ.")
sys.stdout.flush()

# ----------------------------
# 3. MISTRAL FEW-SHOT
# ----------------------------
print("\nChargement de Mistral-7B...")
sys.stdout.flush()
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
    print("Mistral-7B chargé")
except Exception as e:
    mistral_pipe = None
    print("Mistral non disponible :", e)

def predict_with_mistral_balanced(text):
    if mistral_pipe is None:
        return 0, 0.5
    prompt = f"""<s>[INST] Analyse ce commentaire Reddit. Réponds UNIQUEMENT par "SARCASTIQUE" ou "NORMAL".
RÈGLES:
- Ironie, exagération, cynisme → SARCASTIQUE
- Sincère, factuel, neutre → NORMAL
EXEMPLES:"thanks for sharing" → NORMAL
"oh great, another bug" → SARCASTIQUE
Commentaire: "{text[:140]}"
Réponse: [/INST]"""
    try:
        out = mistral_pipe(prompt, max_new_tokens=15, do_sample=False)
        resp = out[0]['generated_text'].split("Réponse:")[-1].strip().upper()
        if "SARCASTIQUE" in resp and "NORMAL" not in resp:
            return 1, 0.92
        elif "NORMAL" in resp and "SARCASTIQUE" not in resp:
            return 0, 0.92
        else:
            return 0, 0.5
    except:
        return 0, 0.5

# ----------------------------
# 4. RÈGLES LINGUISTIQUES
# ----------------------------
def detect_sarcasm_rules(text):
    t = text.lower()
    patterns = [
        r"\boh\s+great\b", r"\bof\s+course\b", r"\bsure[, ]", r"\bobviously\b",
        r"\bthanks?,? i hate", r"\bcool story", r"\bwell played\b",
        r"\bbecause that.?s? never", r"\bi'?m\s+shocked\b", r"\bjust what we needed\b"
    ]
    return 1 if any(re.search(p, t) for p in patterns) else 0

# ----------------------------
# 5. PRÉDICTION DISTILBERT (BATCHÉE)
# ----------------------------
@torch.no_grad()
def get_distilbert_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf, probs

# ----------------------------
# 6. FUSIONS
# ----------------------------
def fusion_parallele(distil_probs, mistral_pred, mistral_conf):
    mistral_probs = np.array([1 - mistral_conf, mistral_conf]) if mistral_pred == 1 else np.array([mistral_conf, 1 - mistral_conf])
    fused = (distil_probs + mistral_probs) / 2
    return int(np.argmax(fused)), float(np.max(fused))

def fusion_sequentielle(distil_pred, distil_conf, mistral_pred, mistral_conf, text):
    rules_pred = detect_sarcasm_rules(text)
    if distil_conf > 0.85:
        return distil_pred, distil_conf, "DistilBERT"
    if mistral_conf > 0.90 and rules_pred == 1:
        return 1, mistral_conf, "Mistral+Règles"
    if distil_pred == mistral_pred and distil_conf > 0.70:
        return distil_pred, (distil_conf + mistral_conf)/2, "Accord"
    return distil_pred, distil_conf, "DistilBERT"

def fusion_intelligente(text):
    d_pred, d_conf, d_probs = get_distilbert_prediction(text)
    m_pred, m_conf = predict_with_mistral_balanced(text) p_pred, p_conf = fusion_parallele(d_probs, m_pred, m_conf)
    s_pred, s_conf, _ = fusion_sequentielle(d_pred, d_conf, m_pred, m_conf, text)
    if d_conf > 0.80:
        return d_pred, d_conf, "DistilBERT"
    elif p_conf > s_conf + 0.05:
        return p_pred, p_conf, "Fusion Parallèle"
    elif s_conf > p_conf + 0.05:
        return s_pred, s_conf, "Fusion Séquentielle"
    else:
        return d_pred, d_conf * 0.9, "DistilBERT (défaut)"

# ----------------------------
# 7. ÉVALUATION SUR 10K
# ----------------------------
test_eval_df = test_df.sample(n=TEST_SIZE, random_state=42).reset_index(drop=True)
print(f"\nÉVALUATION SUR {len(test_eval_df):,} ÉCHANTILLONS DU DATASET TEST\n")
sys.stdout.flush()

start_time = time.time()
y_true = test_eval_df['label'].tolist()
y_pred_distil, y_pred_mistral, y_pred_parallele, y_pred_sequentielle, y_pred_fusion = [], [], [], [], []

print("Évaluation en cours (DistilBERT + Mistral + Fusions)...")
# Création d'une instance tqdm pour un contrôle propre
progress_bar = tqdm(test_eval_df['text'].tolist(), desc="Évaluation", leave=False)
for text in progress_bar:
    # DistilBERT
    d_pred, d_conf, d_probs = get_distilbert_prediction(text)
    # Mistral
    m_pred, m_conf = predict_with_mistral_balanced(text)
    # Fusions
    p_pred, p_conf = fusion_parallele(d_probs, m_pred, m_conf)
    s_pred, s_conf, _ = fusion_sequentielle(d_pred, d_conf, m_pred, m_conf, text)
    f_pred, f_conf, _ = fusion_intelligente(text)
    # Stockage
    y_pred_distil.append(d_pred)
    y_pred_mistral.append(m_pred)
    y_pred_parallele.append(p_pred)
    y_pred_sequentielle.append(s_pred)
    y_pred_fusion.append(f_pred)

# Fermeture propre de la barre de progression
progress_bar.close()

# ----------------------------
# 8. MÉTRIQUES FINALES
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
    "Fusion Intelligente Finale": y_pred_fusion
}

print(f"\n{'='*75}")
print(f"RÉSULTATS FINAUX SUR {len(test_eval_df):,} ÉCHANTILLONS")
print(f"{'Méthode':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print("-" * 75)
for name, pred in methods.items():
    acc, p, r, f1 = metrics(y_true, pred)
    print(f"{name:<32} {acc:10.4f} {p:10.4f} {r:10.4f} {f1:8.4f}")print(f"{'='*75}")

elapsed = time.time() - start_time
h, m = divmod(int(elapsed // 60), 60)
print(f"\nTemps total d'évaluation : {h}h {m}min")
print(f"Évaluation complète sur {len(test_eval_df):,} commentaires terminée.")


===========================================================================
RÉSULTATS FINAUX SUR 10,000 ÉCHANTILLONS
Méthode                            Accuracy  Precision     Recall       F1
---------------------------------------------------------------------------
DistilBERT                           0.7527     0.7368     0.7878   0.7615
Mistral Few-Shot                     0.5904     0.5654     0.7886   0.6586
Fusion Parallèle                     0.6314     0.5917     0.8523   0.6985
Fusion Séquentielle                  0.7500     0.7327     0.7888   0.7597
Fusion Intelligente Finale           0.7465     0.7111     0.8319   0.7668
===========================================================================

Temps total d'évaluation : 2h 18min
Évaluation complète sur 10,000 commentaires terminée.
https://www.kaggle.com/code/maramyaakoubi/notebookbfa95b0b1f?scriptVersionId=277843952