# -*- coding: utf-8 -*-
"""Sarcasm Detection - DistilBERT + Mistral (Few-Shot) + Fusions Complètes"""

import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# CONFIGURATION
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Device: {device}")

# ----------------------------
# 1️⃣ CHARGEMENT DU DATASET
# ----------------------------
def load_sarcasm_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

train_df = load_sarcasm_data('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json')
test_df  = load_sarcasm_data('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json')

# Colonnes cohérentes
train_df = train_df.rename(columns={"headline": "text", "is_sarcastic": "label"})[['text','label']]
test_df  = test_df.rename(columns={"headline": "text", "is_sarcastic": "label"})[['text','label']]

# Nettoyage
train_df['text'] = train_df['text'].astype(str).str.strip()
train_df = train_df[train_df['text'] != ""]
test_df['text'] = test_df['text'].astype(str).str.strip()
test_df = test_df[test_df['text'] != ""]

# ⚡ MODIFICATION : Prendre seulement 50% du dataset de test
test_df = test_df.sample(frac=0.5, random_state=42)
print(f"📊 Taille du dataset de test réduit : {len(test_df)} exemples")

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

# ----------------------------
# 2️⃣ DISTILBERT FINE-TUNING
# ----------------------------
distilbert_model_name = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_name, num_labels=2).to(device)

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_data = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), distilbert_tokenizer)
val_data   = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), distilbert_tokenizer)
test_data  = SarcasmDataset(test_df['text'].tolist(), test_df['label'].tolist(), distilbert_tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./distilbert-news-binary",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=distilbert_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

trainer.train()

# ----------------------------
# 3️⃣ MISTRAL FEW-SHOT AMÉLIORÉ
# ----------------------------
try:
    mistral_pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.1
    )
except:
    mistral_pipe = None

def predict_with_mistral_few_shot(text):
    if mistral_pipe is None:
        return 0, 0.5
    
    prompt = f"""<s>[INST] Analyse ce titre de news et détermine s'il contient du sarcasme.
Réponds UNIQUEMENT par "SARCASTIQUE" ou "NORMAL".

RÈGLES IMPORTANTES:
- Un compliment seul n'est PAS sarcastique  
- Le sarcasme nécessite une CONTRADICTION évidente
- Sois PRUDENT avec les phrases positives

Exemples:
Titre: "thirtysomething scientists unveil doomsday clock of hair loss" → SARCASTIQUE
Titre: "dem rep. totally nails why congress is falling behind" → SARCASTIQUE  
Titre: "eat your veggies: 9 deliciously different recipes" → NORMAL
Titre: "new study shows benefits of regular exercise" → NORMAL
Titre: "this is the best day ever!" → NORMAL (peut être sincère)
Titre: "I love waiting in government offices" → SARCASTIQUE (contradiction)

Titre: "{text[:120]}"
Réponse: [/INST]"""
    
    try:
        outputs = mistral_pipe(prompt, max_new_tokens=20, temperature=0.1, do_sample=True, num_return_sequences=1, pad_token_id=mistral_pipe.tokenizer.eos_token_id)
        response = outputs[0]['generated_text'].split("Réponse:")[-1].strip().upper()
        
        # Vérification stricte
        if "SARCASTIQUE" in response and "NORMAL" not in response:
            return 1, 0.95
        elif "NORMAL" in response and "SARCASTIQUE" not in response:
            return 0, 0.95
        else:
            return 0, 0.5
    except:
        return 0, 0.5

# ----------------------------
# 4️⃣ FUSIONS AMÉLIORÉES
# ----------------------------
def fusion_parallele(distil_probs, mistral_pred, mistral_confidence):
    mistral_probs = np.array([1 - mistral_confidence, mistral_confidence]) if mistral_pred == 1 else np.array([mistral_confidence, 1 - mistral_confidence])
    fused_probs = (distil_probs + mistral_probs) / 2
    return np.argmax(fused_probs), np.max(fused_probs)

def analyser_regles_linguistiques(text):
    """Nouvelle fonction: règles pour détecter le sarcasme"""
    text_lower = text.lower()
    
    # Indicateurs forts de sarcasme
    indicateurs_forts = [
        "yet again", "of course", "sure,", "obviously", "what a surprise",
        "big surprise", "shocked by", "who would have thought", "another"
    ]
    
    # Contradictions évidentes
    contradictions = [
        ("love", "traffic"), ("love", "waiting"), ("love", "paperwork"),
        ("excited", "meeting"), ("great", "monday"), ("fantastic", "broken")
    ]
    
    # Vérifier les indicateurs forts
    if any(indicateur in text_lower for indicateur in indicateurs_forts):
        return 1
    
    # Vérifier les contradictions
    if any(positif in text_lower and negatif in text_lower for positif, negatif in contradictions):
        return 1
    
    return 0

def fusion_sequentielle_amelioree(distil_probs, distil_conf, mistral_pred, mistral_conf, text):
    # Règles linguistiques comme premier critère
    regles_pred = analyser_regles_linguistiques(text)
    
    # Si règles détectent du sarcasme et Mistral aussi → sarcasme
    if regles_pred == 1 and mistral_pred == 1:
        return 1, max(distil_conf, mistral_conf), "Règles + Mistral"
    
    # Logique originale
    if distil_conf > 0.85:
        return np.argmax(distil_probs), distil_conf, "DistilBERT (confiant)"
    if mistral_conf > 0.8:
        return mistral_pred, mistral_conf, "Mistral (confiant)"
    
    sarcasm_indicators = ["obviously", "of course", "sure", "great", "wow", "just what we needed"]
    if any(ind in text.lower() for ind in sarcasm_indicators) and mistral_pred == 1:
        return 1, max(distil_conf, mistral_conf), "Indicateurs + Mistral"
    
    if np.argmax(distil_probs) == mistral_pred:
        return mistral_pred, (distil_conf + mistral_conf)/2, "Accord des modèles"
    
    if len(text.split()) < 8 and mistral_conf > 0.7:
        return mistral_pred, mistral_conf, "Mistral (titre court)"
    
    return np.argmax(distil_probs), distil_conf, "DistilBERT (default)"

def fusion_intelligente_amelioree(text, distilbert_model, distilbert_tokenizer):
    inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(distilbert_model.device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        distil_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        distil_pred = np.argmax(distil_probs)
        distil_conf = np.max(distil_probs)
    
    mistral_pred, mistral_conf = predict_with_mistral_few_shot(text)
    pred_parallele, conf_parallele = fusion_parallele(distil_probs, mistral_pred, mistral_conf)
    pred_sequentielle, conf_sequentielle, methode_seq = fusion_sequentielle_amelioree(distil_probs, distil_conf, mistral_pred, mistral_conf, text)
    
    # NOUVELLE LOGIQUE : Ajout des règles linguistiques dans la décision finale
    regles_pred = analyser_regles_linguistiques(text)
    
    # Si règles détectent du sarcasme fort → favoriser sarcasme
    if regles_pred == 1 and mistral_pred == 1:
        final_pred, final_conf, methode = 1, max(conf_parallele, conf_sequentielle), "Règles + Mistral (fort)"
    elif regles_pred == 1 and conf_parallele > 0.6:
        final_pred, final_conf, methode = 1, conf_parallele, "Règles + Fusion Parallèle"
    elif conf_parallele > conf_sequentielle + 0.1:
        final_pred, final_conf, methode = pred_parallele, conf_parallele, "Fusion Parallèle"
    elif conf_sequentielle > conf_parallele + 0.1:
        final_pred, final_conf, methode = pred_sequentielle, conf_sequentielle, f"Fusion Séquentielle ({methode_seq})"
    else:
        if pred_parallele == pred_sequentielle:
            final_pred, final_conf, methode = pred_parallele, (conf_parallele+conf_sequentielle)/2, "Accord des Fusions"
        else:
            final_pred, final_conf, methode = distil_pred, distil_conf*0.8, "DistilBERT (désaccord résolu)"
    
    return {'final_pred': final_pred, 'final_conf': final_conf, 'methode': methode,
            'distil_pred': distil_pred, 'distil_conf': distil_conf,
            'mistral_pred': mistral_pred, 'mistral_conf': mistral_conf,
            'parallele_pred': pred_parallele, 'parallele_conf': conf_parallele,
            'sequentielle_pred': pred_sequentielle, 'sequentielle_conf': conf_sequentielle}

# ----------------------------
# 5️⃣ TEST DE PHRASES PERSONNALISÉES
# ----------------------------
phrases_test = [
    "Oh great, another Monday morning!",
    "The weather is beautiful today.",
    "I just love getting stuck in traffic.",
    "Local bakery introduces gluten-free options.",
    "Scientists shocked by harmless experiment results!",
    "New study claims chocolate cures all diseases.",
    "Politician promises 'change' yet again.",
]

print("\n===== 🔍 TEST DE PHRASES PERSONNALISÉES =====")
for phrase in phrases_test:
    res = fusion_intelligente_amelioree(phrase, distilbert_model, distilbert_tokenizer)
    distil_label = "Sarcastique" if res['distil_pred']==1 else "Normal"
    mistral_label = "Sarcastique" if res['mistral_pred']==1 else "Normal"
    final_label = "Sarcastique" if res['final_pred']==1 else "Normal"
    print(f"\n📝 Phrase testée : {phrase}")
    print(f"🤖 DistilBERT : {distil_label} (confiance {res['distil_conf']:.2f})")
    print(f"🌟 Mistral : {mistral_label} (confiance {res['mistral_conf']:.2f})")
    print(f"⚡ Fusion Intelligente : {final_label} (confiance {res['final_conf']:.2f})")
    print(f"🔧 Méthode choisie : {res['methode']}")

# ----------------------------
# 6️⃣ ÉVALUATION SUR LA MOITIÉ DU DATASET DE TEST
# ----------------------------
y_true = test_df['label'].tolist()

y_pred_distil = []
y_pred_mistral = []
y_pred_parallele = []
y_pred_sequentielle = []
y_pred_fusion = []

print(f"\n===== 🔍 ÉVALUATION SUR {len(test_df)} EXEMPLES (50% DU DATASET) =====")
for text in tqdm(test_df['text'].tolist(), desc="Évaluation complète"):
    # DISTILBERT
    inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(distilbert_model.device)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_distil = np.argmax(probs)
        conf_distil = np.max(probs)

    # MISTRAL
    pred_mistral, conf_mistral = predict_with_mistral_few_shot(text)

    # FUSIONS
    pred_parallele, conf_parallele = fusion_parallele(probs, pred_mistral, conf_mistral)
    pred_seq, conf_seq, _ = fusion_sequentielle_amelioree(probs, conf_distil, pred_mistral, conf_mistral, text)
    res_fusion = fusion_intelligente_amelioree(text, distilbert_model, distilbert_tokenizer)

    # Sauvegarde
    y_pred_distil.append(pred_distil)
    y_pred_mistral.append(pred_mistral)
    y_pred_parallele.append(pred_parallele)
    y_pred_sequentielle.append(pred_seq)
    y_pred_fusion.append(res_fusion['final_pred'])

# ----------------------------
# 7️⃣ CALCUL DES MÉTRIQUES POUR CHAQUE MÉTHODE
# ----------------------------
def compute_metrics_from_lists(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return acc, precision, recall, f1

methods = {
    "DistilBERT": y_pred_distil,
    "Mistral Few-Shot": y_pred_mistral,
    "Fusion Parallèle": y_pred_parallele,
    "Fusion Séquentielle": y_pred_sequentielle,
    "Fusion Intelligente Finale": y_pred_fusion
}

print(f"\n===== 📊 COMPARAISON DES PERFORMANCES =====")
print(f"{'Méthode':28s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s}")
print("-"*65)
for name, preds in methods.items():
    acc, prec, rec, f1 = compute_metrics_from_lists(y_true, preds)
    print(f"{name:28s} {acc:8.3f} {prec:10.3f} {rec:8.3f} {f1:6.3f}")

# ----------------------------
# 8️⃣ RÉSUMÉ
# ----------------------------
print(f"\n📊 Taille test : {len(y_true)} exemples")
print(f"Taille originale du dataset : {len(load_sarcasm_data('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'))}")
print(f"Pourcentage utilisé : {len(y_true)/len(load_sarcasm_data('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'))*100:.1f}%")



===== 🔍 TEST DE PHRASES PERSONNALISÉES =====

📝 Phrase testée : Oh great, another Monday morning!
🤖 DistilBERT : Normal (confiance 0.99)
🌟 Mistral : Sarcastique (confiance 0.95)
⚡ Fusion Intelligente : Sarcastique (confiance 0.99)
🔧 Méthode choisie : Règles + Mistral (fort)

📝 Phrase testée : The weather is beautiful today.
🤖 DistilBERT : Normal (confiance 1.00)
🌟 Mistral : Normal (confiance 0.95)
⚡ Fusion Intelligente : Normal (confiance 0.99)
🔧 Méthode choisie : Accord des Fusions

📝 Phrase testée : I just love getting stuck in traffic.
🤖 DistilBERT : Normal (confiance 1.00)
🌟 Mistral : Sarcastique (confiance 0.95)
⚡ Fusion Intelligente : Sarcastique (confiance 1.00)
🔧 Méthode choisie : Règles + Mistral (fort)

📝 Phrase testée : Local bakery introduces gluten-free options.
🤖 DistilBERT : Sarcastique (confiance 0.98)
🌟 Mistral : Normal (confiance 0.95)
⚡ Fusion Intelligente : Sarcastique (confiance 0.98)
🔧 Méthode choisie : Fusion Séquentielle (DistilBERT (confiant))

📝 Phrase testée : Scientists shocked by harmless experiment results!
🤖 DistilBERT : Normal (confiance 0.89)
🌟 Mistral : Sarcastique (confiance 0.95)
⚡ Fusion Intelligente : Sarcastique (confiance 0.95)
🔧 Méthode choisie : Règles + Mistral (fort)

📝 Phrase testée : New study claims chocolate cures all diseases.
🤖 DistilBERT : Sarcastique (confiance 0.97)
🌟 Mistral : Sarcastique (confiance 0.95)
⚡ Fusion Intelligente : Sarcastique (confiance 0.96)
🔧 Méthode choisie : Accord des Fusions

📝 Phrase testée : Politician promises 'change' yet again.
🤖 DistilBERT : Normal (confiance 0.99)
🌟 Mistral : Sarcastique (confiance 0.95)
⚡ Fusion Intelligente : Sarcastique (confiance 0.99)
🔧 Méthode choisie : Règles + Mistral (fort)

===== 🔍 ÉVALUATION SUR 13354 EXEMPLES (50% DU DATASET) =====
Error displaying widget: model not found
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

===== 📊 COMPARAISON DES PERFORMANCES =====
Méthode                      Accuracy  Precision   Recall     F1
-----------------------------------------------------------------
DistilBERT                      0.967      0.968    0.957  0.962
Mistral Few-Shot                0.673      0.639    0.585  0.611
Fusion Parallèle                0.944      0.965    0.905  0.934
Fusion Séquentielle             0.959      0.968    0.938  0.953
Fusion Intelligente Finale      0.957      0.964    0.942  0.951

📊 Taille test : 13354 exemples
Taille originale du dataset : 26709
Pourcentage utilisé : 50.0%






https://www.kaggle.com/code/dalandamelki/notebookd362f0d710/edit