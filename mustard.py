# -*- coding: utf-8 -*-
"""Sarcasm Detection OPTIMISÉ - DistilBERT + Mistral + Fusions COMPLÈTES - MUSTARD Dataset - Version Finale Améliorée"""

import os
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

# ----------------------------
# CONFIGURATION
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Device: {device}")

# ----------------------------
# 1️⃣ CHARGEMENT ET PRÉPARATION DES DONNÉES
# ----------------------------
def load_mustard_dataset(file_path):
    """Charge le dataset MUSTARD"""
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
    print(f"📊 MUSTARD Dataset: {len(df)} exemples")
    print(f"🎯 Distribution: {df['label'].value_counts().to_dict()}")
    
    return df

# Chargement des données
mustard_path = '/kaggle/input/mustard-multimodal-sarcasm-detection-dataset/sarcasm_data.json'
df = load_mustard_dataset(mustard_path)

# Split train/val/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, stratify=train_df['label'])

print(f"\n📊 Split des données:")
print(f"   Train: {len(train_df)} exemples")
print(f"   Validation: {len(val_df)} exemples") 
print(f"   Test: {len(test_df)} exemples")

# ----------------------------
# 2️⃣ DISTILBERT FINE-TUNING OPTIMISÉ
# ----------------------------
distilbert_model_name = "distilbert-base-uncased"

# Configuration du tokenizer
tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✅ Tokenizer configuré - Pad token: {tokenizer.pad_token}")

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

# Préparation des datasets
train_dataset = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
test_dataset = SarcasmDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

# Initialisation du modèle
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

# Configuration d'entraînement améliorée
training_args = TrainingArguments(
    output_dir="./distilbert-mustard-v2",
    num_train_epochs=6,  # Plus d'epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,  # Learning rate plus bas
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
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("🚀 Entraînement de DistilBERT...")
trainer.train()

# Sauvegarde du modèle
trainer.save_model("./distilbert-mustard-v2")
print("✅ DistilBERT entraîné et sauvegardé!")

# ----------------------------
# 3️⃣ MISTRAL 7B - VERSION OPTIMISÉE
# ----------------------------
print("🔄 Chargement de Mistral 7B...")
try:
    mistral_pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
        torch_dtype=torch.float16,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    print("✅ Mistral 7B chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur chargement Mistral: {e}")
    print("🔄 Tentative de chargement avec configuration alternative...")
    try:
        mistral_pipe = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16,
            max_new_tokens=50
        )
        print("✅ Mistral 7B chargé avec configuration alternative!")
    except:
        mistral_pipe = None
        print("⚠️ Mistral non disponible - utilisation des autres méthodes")

def predict_with_mistral_optimized(text):
    """Version optimisée de Mistral avec prompt amélioré"""
    if mistral_pipe is None:
        return 0, 0.5
    
    # Prompt optimisé pour MUSTARD
    prompt = f"""<s>[INST] Analyse cette phrase et détermine si elle est sarcastique. 
Réponds UNIQUEMENT par "SARCASTIQUE" ou "NORMAL".

Phrase: "{text}"

Réponse: [/INST]"""
    
    try:
        outputs = mistral_pipe(
            prompt,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = outputs[0]['generated_text']
        
        # Extraction robuste de la réponse
        if "Réponse:" in response:
            answer = response.split("Réponse:")[-1].strip()
        else:
            answer = response[-50:].strip()
        
        answer_clean = answer.upper()
        
        # Détection robuste
        if "SARCASTIQUE" in answer_clean and "NORMAL" not in answer_clean:
            confidence = 0.9
            return 1, confidence
        elif "NORMAL" in answer_clean and "SARCASTIQUE" not in answer_clean:
            confidence = 0.9
            return 0, confidence
        else:
            # Si ambigu, analyse basique
            sarcasm_indicators = ["great", "wonderful", "perfect", "love", "sure", "obviously", "of course"]
            if any(indicator in text.lower() for indicator in sarcasm_indicators):
                return 1, 0.6
            else:
                return 0, 0.5
                
    except Exception as e:
        print(f"⚠️ Erreur Mistral: {e}")
        return 0, 0.5

# ----------------------------
# 4️⃣ RÈGLES LINGUISTIQUES AMÉLIORÉES - SPÉCIFIQUES MUSTARD
# ----------------------------
def analyser_regles_linguistiques_ameliore(text):
    """Règles spécifiques à MUSTARD basées sur l'analyse du dataset"""
    text_lower = text.lower()
    score = 0
    
    # Patterns spécifiques à MUSTARD
    patterns_mustard = {
        # Sarcasme par exagération
        "love how": 2, "so excited": 2, "big surprise": 3, "of course": 3,
        "oh great": 3, "wonderful": 2, "perfect": 2, "fantastic": 2,
        "really": 1, "sure": 2, "obviously": 2, "clearly": 2,
        "what a": 2, "how": 1, "finally": 1, "again": 1, "another": 2,
        "as always": 2, "right on time": 2, "my favorite": 2
    }
    
    # Contradictions détectées dans MUSTARD
    contradictions = [
        ("love", "hate"), ("great", "terrible"), ("wonderful", "awful"),
        ("perfect", "mess"), ("excited", "bored"), ("best", "worst"),
        ("thrilled", "disappointed"), ("fantastic", "horrible")
    ]
    
    # Calcul du score amélioré
    for pattern, points in patterns_mustard.items():
        if pattern in text_lower:
            score += points
    
    # Vérification des contradictions
    for pos, neg in contradictions:
        if pos in text_lower and any(neg_word in text_lower for neg_word in [neg, "not " + pos, "never " + pos]):
            score += 2
    
    # Analyse de la structure
    if "!" in text and any(word in text_lower for word in ["great", "wonderful", "perfect", "fantastic", "love"]):
        score += 2
    
    # Ton interrogatif sarcastique
    if "?" in text and any(word in text_lower for word in ["great", "wonderful", "perfect"]):
        score += 1
    
    # Seuil adaptatif basé sur la longueur
    word_count = len(text.split())
    threshold = 2 if word_count > 3 else 1.5
    
    return 1 if score >= threshold else 0, score

# ----------------------------
# 5️⃣ SYSTÈME DE FUSION COMPLET - TOUTES LES MÉTHODES IMPLÉMENTÉES
# ----------------------------

def fusion_parallele_optimale(distil_probs, mistral_pred, mistral_conf, regles_score):
    """Fusion parallèle avec pondération intelligente"""
    # Conversion des probabilités Mistral
    if mistral_pred == 1:
        mistral_probs = np.array([1 - mistral_conf, mistral_conf])
    else:
        mistral_probs = np.array([mistral_conf, 1 - mistral_conf])
    
    # Pondération basée sur la confiance et les performances
    distil_weight = 1.0  # DistilBERT bien performant
    mistral_weight = mistral_conf * 1.3  # Mistral modérément pondéré
    
    # Boost modéré si règles détectent du sarcasme fort
    if regles_score >= 3:
        mistral_weight += 0.3
    
    # Fusion pondérée
    total_weight = distil_weight + mistral_weight
    fused_probs = (distil_probs * distil_weight + mistral_probs * mistral_weight) / total_weight
    
    return np.argmax(fused_probs), np.max(fused_probs)

def fusion_sequentielle_amelioree(distil_probs, distil_conf, mistral_pred, mistral_conf, text):
    """🔥 FONCTION SÉQUENTIELLE MANQUANTE - MAINTENANT IMPLÉMENTÉE"""
    regles_pred, regles_score = analyser_regles_linguistiques_ameliore(text)
    
    # 1. Si règles détectent fort sarcasme ET Mistral confirme
    if regles_score >= 4 and mistral_pred == 1:
        return 1, max(distil_conf, mistral_conf), "Règles Fortes + Mistral"
    
    # 2. Si DistilBERT très confiant
    if distil_conf > 0.85:
        return np.argmax(distil_probs), distil_conf, "DistilBERT (très confiant)"
    
    # 3. Si Mistral très confiant
    if mistral_conf > 0.85:
        return mistral_pred, mistral_conf, "Mistral (très confiant)"
    
    # 4. Si règles détectent sarcasme et au moins un modèle confirme
    if regles_pred == 1 and (mistral_pred == 1 or np.argmax(distil_probs) == 1):
        confidence_boost = 0.1 if regles_score >= 3 else 0.05
        return 1, max(distil_conf, mistral_conf) + confidence_boost, "Règles + Modèle"
    
    # 5. Si les deux modèles sont d'accord
    if np.argmax(distil_probs) == mistral_pred:
        combined_conf = (distil_conf + mistral_conf) / 2
        return mistral_pred, combined_conf, "Accord des modèles"
    
    # 6. Par défaut, suivre DistilBERT
    return np.argmax(distil_probs), distil_conf * 0.9, "DistilBERT (défaut)"

def fusion_intelligente_reequilibree(text, distilbert_model, distilbert_tokenizer):
    """Fusion rééquilibrée utilisant TOUTES les méthodes"""
    
    # 1. DISTILBERT
    inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(distilbert_model.device) for k,v in inputs.items()}
    
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        distil_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        distil_pred = np.argmax(distil_probs)
        distil_conf = np.max(distil_probs)
    
    # 2. MISTRAL
    mistral_pred, mistral_conf = predict_with_mistral_optimized(text)
    
    # 3. RÈGLES LINGUISTIQUES AMÉLIORÉES
    regles_pred, regles_score = analyser_regles_linguistiques_ameliore(text)
    
    # 4. 🔥 FUSION SÉQUENTIELLE (NOUVELLE)
    seq_pred, seq_conf, seq_method = fusion_sequentielle_amelioree(
        distil_probs, distil_conf, mistral_pred, mistral_conf, text
    )
    
    # 5. FUSION PARALLÈLE
    par_pred, par_conf = fusion_parallele_optimale(distil_probs, mistral_pred, mistral_conf, regles_score)
    
    # 6. FUSION INTELLIGENTE AVEC TOUTES LES MÉTHODES
    weights = {
        'distilbert': 0.50,   # Poids réduit pour laisser place aux autres
        'mistral': 0.25,      # Poids modéré
        'regles': 0.10,       # Poids pour règles
        'sequentielle': 0.15  # 🔥 NOUVEAU: poids pour fusion séquentielle
    }
    
    # Conversion des probabilités
    if mistral_pred == 1:
        mistral_probs = np.array([1 - mistral_conf, mistral_conf])
    else:
        mistral_probs = np.array([mistral_conf, 1 - mistral_conf])
    
    # Probabilités règles
    regles_probs = np.array([0.5, 0.5])
    if regles_score >= 4:
        regles_probs = np.array([0.1, 0.9])
    elif regles_score >= 2:
        regles_probs = np.array([0.3, 0.7])
    elif regles_score >= 1:
        regles_probs = np.array([0.4, 0.6])
    
    # 🔥 Probabilités fusion séquentielle
    if seq_pred == 1:
        seq_probs = np.array([1 - seq_conf, seq_conf])
    else:
        seq_probs = np.array([seq_conf, 1 - seq_conf])
    
    # Fusion finale avec TOUTES les méthodes
    final_probs = (
        distil_probs * weights['distilbert'] +
        mistral_probs * weights['mistral'] +
        regles_probs * weights['regles'] +
        seq_probs * weights['sequentielle']  # 🔥 AJOUT de la fusion séquentielle
    )
    
    final_pred = np.argmax(final_probs)
    final_conf = np.max(final_probs)
    
    # Détection de la méthode dominante
    method_contributions = {
        'DistilBERT': weights['distilbert'] * distil_probs[final_pred],
        'Mistral': weights['mistral'] * mistral_probs[final_pred],
        'Règles': weights['regles'] * regles_probs[final_pred],
        'Fusion Séquentielle': weights['sequentielle'] * seq_probs[final_pred]  # 🔥 NOUVEAU
    }
    
    dominant_method = max(method_contributions.items(), key=lambda x: x[1])
    
    return {
        'final_pred': final_pred, 'final_conf': final_conf, 'methode': f"Fusion - {dominant_method[0]} dominant",
        'distil_pred': distil_pred, 'distil_conf': distil_conf,
        'mistral_pred': mistral_pred, 'mistral_conf': mistral_conf,
        'regles_pred': regles_pred, 'regles_score': regles_score,
        'sequentielle_pred': seq_pred, 'sequentielle_conf': seq_conf, 'sequentielle_methode': seq_method,  # 🔥 NOUVEAU
        'parallele_pred': par_pred, 'parallele_conf': par_conf,  # 🔥 NOUVEAU
        'final_probs': final_probs,
        'method_contributions': method_contributions  # 🔥 NOUVEAU: voir les contributions
    }

# ----------------------------
# 6️⃣ ÉVALUATION COMPLÈTE SUR LE DATASET DE TEST
# ----------------------------
print(f"\n===== 🔍 ÉVALUATION SUR {len(test_df)} EXEMPLES MUSTARD =====")

y_true = test_df['label'].tolist()

# Initialisation des prédictions - AVEC TOUTES LES MÉTHODES
predictions = {
    "DistilBERT": [],
    "Mistral": [],
    "Règles Linguistiques": [],
    "Fusion Parallèle": [],
    "Fusion Séquentielle": [],  # 🔥 NOUVELLE MÉTHODE
    "Fusion Intelligente Complète": []  # 🔥 RENOMMÉ pour refléter l'ajout
}

# Calcul des prédictions avec gestion d'erreurs
success_count = 0
for i, (text, true_label) in enumerate(tqdm(zip(test_df['text'].tolist(), y_true), desc="Évaluation", total=len(test_df))):
    try:
        # DistilBERT seul
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            distil_pred = np.argmax(probs)
        
        # Mistral seul
        mistral_pred, mistral_conf = predict_with_mistral_optimized(text)
        
        # Règles seules
        regles_pred, regles_score = analyser_regles_linguistiques_ameliore(text)
        
        # Fusion parallèle
        fusion_par_pred, fusion_par_conf = fusion_parallele_optimale(probs, mistral_pred, mistral_conf, regles_score)
        
        # 🔥 FUSION SÉQUENTIELLE (NOUVELLE)
        fusion_seq_pred, fusion_seq_conf, fusion_seq_method = fusion_sequentielle_amelioree(
            probs, np.max(probs), mistral_pred, mistral_conf, text
        )
        
        # Fusion intelligente complète (avec séquentielle)
        fusion_data = fusion_intelligente_reequilibree(text, model, tokenizer)
        
        # Stockage des résultats
        predictions["DistilBERT"].append(distil_pred)
        predictions["Mistral"].append(mistral_pred)
        predictions["Règles Linguistiques"].append(regles_pred)
        predictions["Fusion Parallèle"].append(fusion_par_pred)
        predictions["Fusion Séquentielle"].append(fusion_seq_pred)  # 🔥 NOUVEAU
        predictions["Fusion Intelligente Complète"].append(fusion_data['final_pred'])
        
        success_count += 1
        
    except Exception as e:
        print(f"❌ Erreur sur l'échantillon {i}: {e}")
        # Valeurs par défaut en cas d'erreur
        for key in predictions:
            predictions[key].append(0)

print(f"✅ {success_count}/{len(test_df)} prédictions réussies")

# ----------------------------
# 7️⃣ CALCUL DES MÉTRIQUES DÉTAILLÉES
# ----------------------------
def compute_comprehensive_metrics(y_true, y_pred, method_name):
    """Calcule les métriques complètes"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Rapport de classification détaillé
    report = classification_report(y_true, y_pred, target_names=['Non-Sarcastique', 'Sarcastique'], output_dict=True)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

print(f"\n===== 📊 PERFORMANCES DÉTAILLÉES - TOUTES LES MÉTHODES =====")
print(f"{'Méthode':40s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s}")
print("-" * 80)

results = {}
for method_name, y_pred in predictions.items():
    metrics = compute_comprehensive_metrics(y_true, y_pred, method_name)
    results[method_name] = metrics
    print(f"{method_name:40s} {metrics['accuracy']:8.3f} {metrics['precision']:10.3f} {metrics['recall']:8.3f} {metrics['f1']:6.3f}")

# ----------------------------
# 8️⃣ TESTS AVEC EXEMPLES DÉMONSTRATIFS AMÉLIORÉS
# ----------------------------
test_examples = [
    "Oh great, another Monday morning!",
    "The weather is beautiful today.",
    "I just love getting stuck in traffic for hours.", 
    "Can you pass me the salt please?",
    "Wow, you're really helpful with that suggestion.",
    "This is exactly what I needed right now.",
    "What a surprise, you're late again.",
    "I'm so thrilled to be doing paperwork on a Friday night.",
    "Of course, the one time I need help, no one is around.",
    "This is a very useful scientific tool for research.",
    "Love how my computer crashes during important work.",
    "Another brilliant idea from the management team.",
    "Perfect timing as always.",
    "I'm so excited to clean the entire house on my day off."
]

print(f"\n===== 🔍 TESTS AVEC EXEMPLES DÉMONSTRATIFS - TOUTES LES FUSIONS =====")
for i, phrase in enumerate(test_examples, 1):
    try:
        result = fusion_intelligente_reequilibree(phrase, model, tokenizer)
        final_label = "SARCASME" if result['final_pred'] == 1 else "NORMAL"
        
        print(f"\n{i}. 📝 '{phrase}'")
        print(f"   🎯 Résultat final: {final_label} (confiance: {result['final_conf']:.2f})")
        print(f"   🔧 Méthode utilisée: {result['methode']}")
        print(f"   🤖 DistilBERT: {'SARCASME' if result['distil_pred']==1 else 'NORMAL'} ({result['distil_conf']:.2f})")
        print(f"   🌟 Mistral: {'SARCASME' if result['mistral_pred']==1 else 'NORMAL'} ({result['mistral_conf']:.2f})")
        print(f"   📚 Règles: {'SARCASME' if result['regles_pred']==1 else 'NORMAL'} (score: {result['regles_score']})")
        print(f"   🔄 Séquentielle: {'SARCASME' if result['sequentielle_pred']==1 else 'NORMAL'} ({result['sequentielle_conf']:.2f}) - {result['sequentielle_methode']}")
        print(f"   ⚖️ Parallèle: {'SARCASME' if result['parallele_pred']==1 else 'NORMAL'} ({result['parallele_conf']:.2f})")
        print(f"   📊 Probabilités finales: [Non-Sarc: {result['final_probs'][0]:.3f}, Sarc: {result['final_probs'][1]:.3f}]")
        
        # Afficher les contributions des méthodes
        print(f"   📈 Contributions: {result['method_contributions']}")
        
    except Exception as e:
        print(f"❌ Erreur sur l'exemple {i}: {e}")

# ----------------------------
# 9️⃣ ANALYSE DES PERFORMANCES ET RAPPORT FINAL
# ----------------------------
# Trouver la meilleure méthode
best_method = max(results.items(), key=lambda x: x[1]['f1'])

print(f"\n===== ✅ RAPPORT FINAL COMPLET - TOUTES LES FUSIONS =====")
print(f"🏆 MEILLEURE MÉTHODE: {best_method[0]}")
print(f"   F1-score: {best_method[1]['f1']:.3f}")
print(f"   Accuracy: {best_method[1]['accuracy']:.3f}")
print(f"   Precision: {best_method[1]['precision']:.3f}")
print(f"   Recall: {best_method[1]['recall']:.3f}")

print(f"\n📊 COMPARAISON COMPLÈTE DES MÉTHODES (par F1-score):")
for method_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
    print(f"   {method_name:40s} F1: {metrics['f1']:.3f} | Acc: {metrics['accuracy']:.3f}")

print(f"\n🔧 FUSIONS IMPLÉMENTÉES:")
print(f"   ✅ Fusion Parallèle: Combinaison linéaire des probabilités")
print(f"   ✅ Fusion Séquentielle: Décision hiérarchique basée sur la confiance")  
print(f"   ✅ Fusion Intelligente Complète: Combinaison pondérée de toutes les méthodes")
print(f"   ✅ Règles Linguistiques: Patterns spécifiques MUSTARD avec scoring")

print(f"\n📈 STATISTIQUES:")
print(f"   Dataset MUSTARD: {len(df)} échantillons")
print(f"   Méthodes évaluées: {len(results)}")
print(f"   Prédictions réussies: {success_count}/{len(test_df)}")

print(f"\n🎯 RECOMMANDATIONS FINALES:")
print(f"   • '{best_method[0]}' offre les meilleures performances")
print(f"   • La fusion séquentielle ajoute une logique décisionnelle hiérarchique")
print(f"   • La fusion intelligente combine le meilleur de toutes les approches")

print(f"\n🚀 SYSTÈME COMPLET AVEC TOUTES LES FUSIONS PRÊT POUR LA DÉTECTION DE SARCASME!")



✅ 138/138 prédictions réussies

===== 📊 PERFORMANCES DÉTAILLÉES - TOUTES LES MÉTHODES =====
Méthode                                  Accuracy  Precision   Recall     F1
--------------------------------------------------------------------------------
DistilBERT                                  0.587      0.556    0.870  0.678
Mistral                                     0.609      0.674    0.420  0.518
Règles Linguistiques                        0.514      0.583    0.101  0.173
Fusion Parallèle                            0.638      0.634    0.652  0.643
Fusion Séquentielle                         0.638      0.634    0.652  0.643
Fusion Intelligente Complète                0.638      0.634    0.652  0.643


https://www.kaggle.com/code/ahmedyk/notebook9283cb0ec8/edit