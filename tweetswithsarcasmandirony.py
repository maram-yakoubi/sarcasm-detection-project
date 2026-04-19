# -*- coding: utf-8 -*-
"""Sarcasm Detection with Dual Fusion - Tweets Dataset (COMPLETE TESTING)"""

# Installations nécessaires
!pip install transformers datasets evaluate accelerate tqdm --quiet

import os
import torch
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, pipeline
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Device: {device}")

# ----------------------------
# 1️⃣ CHARGEMENT ET NETTOYAGE - ADAPTÉ POUR LES 4 CLASSES EN BINAIRE
# ----------------------------

print("🔹 Étape 1: Chargement et conversion des 4 classes en binaire...")

# Charger les données
train_df = pd.read_csv('/kaggle/input/tweets-with-sarcasm-and-irony/train.csv')
test_df = pd.read_csv('/kaggle/input/tweets-with-sarcasm-and-irony/test.csv')

# Renommer les colonnes
train_df = train_df.rename(columns={"tweets": "text"})
test_df = test_df.rename(columns={"tweets": "text"})

# ✅ MAPPING BINAIRE (figurative/irony/sarcasm → 1, regular → 0)
label_mapping = {
    "figurative": 1,  # Langage figuré = sarcastique
    "irony": 1,       # Ironie = sarcastique  
    "sarcasm": 1,     # Sarcasme = sarcastique
    "regular": 0      # Langage normal = non sarcastique
}

# Appliquer le mapping
train_df['label'] = train_df['class'].map(label_mapping)
test_df['label'] = test_df['class'].map(label_mapping)

# Vérifier la conversion
print("📊 Distribution originale (train):")
print(train_df['class'].value_counts())
print("\n📊 Conversion binaire (train):")
print(train_df['label'].value_counts())
print("→ 0 = Regular (normal), 1 = Figurative/Irony/Sarcasm")

# Nettoyage des données
train_df = train_df.dropna(subset=['text', 'label'])
test_df = test_df.dropna(subset=['text', 'label'])

train_df['text'] = train_df['text'].astype(str).str.strip()
train_df = train_df[train_df['text'] != ""]

test_df['text'] = test_df['text'].astype(str).str.strip()
test_df = test_df[test_df['text'] != ""]

print(f"📊 Train size: {len(train_df)}")
print(f"📊 Test size COMPLET: {len(test_df)}")

# ----------------------------
# 2️⃣ SPLIT DES DONNÉES
# ----------------------------

print("🔹 Étape 2: Split train/val...")

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test COMPLET: {len(test_df)}")
print(f"📊 Distribution test complet: {test_df['label'].value_counts()}")

# ----------------------------
# 3️⃣ FINE-TUNING DISTILBERT - BINAIRE
# ----------------------------

print("🔹 Étape 3: Fine-tuning de DistilBERT (classification binaire)...")

# Configuration DistilBERT
distilbert_model_name = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    distilbert_model_name, 
    num_labels=2  # ⚠️ IMPORTANT: 2 classes pour binaire
).to(device)

# Dataset class
class SarcasmDataset(torch.utils.data.Dataset):
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

# Préparation des datasets
train_data = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), distilbert_tokenizer)
val_data = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), distilbert_tokenizer)
test_data = SarcasmDataset(test_df['text'].tolist(), test_df['label'].tolist(), distilbert_tokenizer)

# Métriques
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./distilbert-tweets-binary-complete",
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

# Trainer
trainer = Trainer(
    model=distilbert_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Entraînement
print("🔹 Début de l'entraînement DistilBERT (classification binaire)...")
trainer.train()

# Évaluation sur le test set COMPLET
print("🔹 Évaluation sur le test set COMPLET...")
distilbert_results = trainer.evaluate(test_data)
print("📊 Résultats DistilBERT sur TOUS les tweets:", distilbert_results)

# ----------------------------
# 4️⃣ FONCTIONS DE FUSION INTELLIGENTES - AVEC MISTRAL FEW-SHOT
# ----------------------------

print("🔹 Étape 4: Configuration des fusions intelligentes avec Mistral few-shot...")

# Charger Mistral
try:
    mistral_pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16,
        max_new_tokens=10,
        do_sample=False
    )
    print("✅ Mistral chargé avec succès")
except Exception as e:
    print(f"⚠️  Impossible de charger Mistral: {e}")
    mistral_pipe = None

def predict_with_mistral_few_shot(text):
    """Prédiction avec Mistral en few-shot learning pour les tweets"""
    if mistral_pipe is None:
        return 0, 0.5
    
    # Prompt few-shot avec exemples concrets de tweets
    prompt = f"""<s>[INST] Analyse ces tweets et détermine s'ils contiennent du sarcasme, de l'ironie ou du langage figuré.
Réponds UNIQUEMENT par "SARCASTIQUE" ou "NON_SARCASTIQUE".

Exemples:
Tweet: "Oh great, another Monday. Just what I needed." → SARCASTIQUE
Tweet: "I love waiting in line for 2 hours. So much fun!" → SARCASTIQUE  
Tweet: "The weather is beautiful today." → NON_SARCASTIQUE
Tweet: "My phone battery died at 10%. How convenient." → SARCASTIQUE
Tweet: "I just finished all my work on time." → NON_SARCASTIQUE
Tweet: "Another perfect day with no surprises at all." → SARCASTIQUE
Tweet: "The package arrived exactly when they said it would." → NON_SARCASTIQUE

Maintenant analyse ce tweet:
Tweet: "{text[:150]}"

Réponse: [/INST]"""
    
    try:
        outputs = mistral_pipe(
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=mistral_pipe.tokenizer.eos_token_id,
        )
        
        full_response = outputs[0]['generated_text']
        response_part = full_response.split("Réponse:")[-1].strip()
        cleaned_response = response_part.upper().replace('.', '').replace('"', '').replace("'", "").strip()
        
        # Détection améliorée des réponses
        if "SARCASTIQUE" in cleaned_response and "NON_SARCASTIQUE" not in cleaned_response:
            return 1, 0.9
        elif "NON_SARCASTIQUE" in cleaned_response:
            return 0, 0.9
        elif "SARCASTIQUE" in cleaned_response:
            return 1, 0.9
        else:
            # Fallback basé sur des mots-clés si la réponse n'est pas claire
            sarcasm_keywords = ["great", "love", "perfect", "obviously", "of course", "sure", "wow", "😂", "😏", "/s"]
            if any(keyword in text.lower() for keyword in sarcasm_keywords):
                return 1, 0.7
            else:
                return 0, 0.5
                
    except Exception as e:
        print(f"❌ Erreur Mistral: {e}")
        # Fallback basé sur des motifs courants de sarcasme
        sarcasm_patterns = [
            r"\b(great|love|perfect|obviously|of course|sure|wow)\b.*[!?]",
            r"😂|😏|/s",
            r"\bjust what i needed\b",
            r"\bso much fun\b"
        ]
        if any(re.search(pattern, text.lower()) for pattern in sarcasm_patterns):
            return 1, 0.6
        return 0, 0.5

def fusion_parallele(distil_probs, mistral_pred, mistral_confidence):
    mistral_probs = np.array([1 - mistral_confidence, mistral_confidence]) 
    if mistral_pred == 0:
        mistral_probs = np.array([mistral_confidence, 1 - mistral_confidence])
    fused_probs = (distil_probs + mistral_probs) / 2
    return np.argmax(fused_probs), np.max(fused_probs)

def fusion_sequentielle(distil_probs, distil_confidence, mistral_pred, mistral_confidence, text):
    if distil_confidence > 0.95:
        return np.argmax(distil_probs), distil_confidence, "DistilBERT (très confiant)"
    
    if mistral_confidence > 0.9:
        return mistral_pred, mistral_confidence, "Mistral (très confiant)"
    
    # Indicateurs de sarcasme adaptés pour les tweets (inclut les 4 classes)
    sarcasm_indicators = [
        "#sarcasm", "#irony", "#figurative", "obviously", "of course", "sure", 
        "definitely", "wow", "great", "😂", "😏", "/s", "sarcastic", "ironic"
    ]
    has_indicator = any(indicator in text.lower() for indicator in sarcasm_indicators)
    
    if has_indicator and mistral_pred == 1:
        return 1, max(distil_confidence, mistral_confidence), "Indicateurs + Mistral"
    
    if np.argmax(distil_probs) == mistral_pred:
        return mistral_pred, (distil_confidence + mistral_confidence) / 2, "Accord des modèles"
    
    return np.argmax(distil_probs), distil_confidence, "DistilBERT (default)"

def fusion_intelligente(text, distilbert_model, distilbert_tokenizer):
    inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(distilbert_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        distil_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        distil_pred = np.argmax(distil_probs)
        distil_confidence = np.max(distil_probs)
    
    # Utiliser la nouvelle fonction few-shot
    mistral_pred, mistral_confidence = predict_with_mistral_few_shot(text)
    
    pred_parallele, conf_parallele = fusion_parallele(distil_probs, mistral_pred, mistral_confidence)
    pred_sequentielle, conf_sequentielle, methode_seq = fusion_sequentielle(
        distil_probs, distil_confidence, mistral_pred, mistral_confidence, text
    )
    
    if conf_parallele > conf_sequentielle:
        final_pred = pred_parallele
        final_conf = conf_parallele
        methode = "Fusion Parallèle"
    else:
        final_pred = pred_sequentielle
        final_conf = conf_sequentielle
        methode = f"Fusion Séquentielle ({methode_seq})"
    
    return {
        'final_pred': final_pred,
        'final_conf': final_conf,
        'methode': methode,
        'distil_pred': distil_pred,
        'distil_conf': distil_confidence,
        'mistral_pred': mistral_pred,
        'mistral_conf': mistral_confidence,
        'parallele_pred': pred_parallele,
        'parallele_conf': conf_parallele,
        'sequentielle_pred': pred_sequentielle,
        'sequentielle_conf': conf_sequentielle
    }

# ----------------------------
# 5️⃣ ÉVALUATION SUR LE TEST SET COMPLET
# ----------------------------

print("🔹 Étape 5: Évaluation des fusions intelligentes sur TOUT le dataset de test...")

# Utiliser TOUT le dataset de test
test_complete = test_df
y_true = test_complete['label'].tolist()
results = []

print(f"🔹 Test sur TOUS les {len(test_complete)} tweets...")
print(f"📊 Distribution complète des labels: {pd.Series(y_true).value_counts()}")

# Prédictions avec barre de progression
for i, text in enumerate(tqdm(test_complete['text'].tolist(), desc="Prédictions sur dataset complet")):
    try:
        result = fusion_intelligente(text, distilbert_model, distilbert_tokenizer)
        result['true_label'] = y_true[i]
        result['text'] = text
        result['original_class'] = test_complete.iloc[i]['class']
        results.append(result)
        
        # Afficher la progression tous les 500 tweets
        if (i + 1) % 500 == 0:
            print(f"✅ {i + 1} tweets traités...")
        
    except Exception as e:
        print(f"❌ Erreur sur le tweet {i}: {e}")
        # Fallback vers DistilBERT seul
        inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(distilbert_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            distil_pred = torch.argmax(outputs.logits).cpu().item()
            distil_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            distil_confidence = np.max(distil_probs)
        
        results.append({
            'final_pred': distil_pred,
            'final_conf': distil_confidence,
            'methode': "Fallback (DistilBERT)",
            'true_label': y_true[i],
            'text': text,
            'original_class': test_complete.iloc[i]['class'],
            'distil_pred': distil_pred,
            'distil_conf': distil_confidence,
            'mistral_pred': -1,  # Indique une erreur
            'mistral_conf': 0.0
        })

# ----------------------------
# 6️⃣ ANALYSE COMPARATIVE COMPLÈTE
# ----------------------------

print("🔹 Étape 6: Analyse comparative complète...")

# Convertir en DataFrame pour analyse
results_df = pd.DataFrame(results)

# Extraire les prédictions
y_pred_distil = [r['distil_pred'] for r in results if 'distil_pred' in r and r['distil_pred'] != -1]
y_pred_mistral = [r['mistral_pred'] for r in results if 'mistral_pred' in r and r['mistral_pred'] != -1]
y_pred_fusion = [r['final_pred'] for r in results]
y_pred_parallele = [r['parallele_pred'] for r in results if 'parallele_pred' in r]
y_pred_sequentielle = [r['sequentielle_pred'] for r in results if 'sequentielle_pred' in r]

# Métriques complètes
def calculate_complete_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Rapport de classification détaillé
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'Class_Report': class_report,
        'Confusion_Matrix': cm
    }

# Comparaison complète
comparison_complete = []
comparison_complete.append(calculate_complete_metrics(y_true, y_pred_distil, "DistilBERT"))
comparison_complete.append(calculate_complete_metrics(y_true, y_pred_mistral, "Mistral-7B (Few-Shot)"))
comparison_complete.append(calculate_complete_metrics(y_true, y_pred_parallele, "Fusion Parallèle"))
comparison_complete.append(calculate_complete_metrics(y_true, y_pred_sequentielle, "Fusion Séquentielle"))
comparison_complete.append(calculate_complete_metrics(y_true, y_pred_fusion, "Fusion Intelligente"))

# Affichage des résultats
print("\n" + "="*100)
print("📊 COMPARAISON COMPLÈTE SUR TOUT LE DATASET - 8,119 TWEETS")
print("="*100)

# Tableau comparatif
comparison_df = pd.DataFrame([{k: v for k, v in item.items() if k != 'Class_Report' and k != 'Confusion_Matrix'} 
                             for item in comparison_complete])
print(comparison_df.to_string(index=False))

# ----------------------------
# 7️⃣ VISUALISATIONS ET ANALYSES DÉTAILLÉES
# ----------------------------

print("\n🔹 Étape 7: Visualisations et analyses détaillées...")

# 1. Répartition des méthodes de fusion
method_counts = results_df['methode'].value_counts()
print(f"\n🔧 RÉPARTITION DES MÉTHODES ({len(results_df)} prédictions):")
for method, count in method_counts.items():
    percentage = (count / len(results_df)) * 100
    print(f"  {method}: {count} ({percentage:.2f}%)")

# 2. Analyse par classe originale
print(f"\n📊 PERFORMANCE PAR CLASSE ORIGINALE:")
class_performance = results_df.groupby('original_class').apply(
    lambda x: accuracy_score(x['true_label'], x['final_pred'])
)
print(class_performance)

# 3. Matrice de confusion pour la fusion intelligente
final_cm = confusion_matrix(y_true, y_pred_fusion)
plt.figure(figsize=(10, 8))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Regular', 'Sarcastic'], 
            yticklabels=['Regular', 'Sarcastic'])
plt.title('Matrice de Confusion - Fusion Intelligente\n(Dataset Complet: 8,119 tweets)')
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes')
plt.show()

# 4. Distribution des confiances
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist([r['distil_conf'] for r in results if 'distil_conf' in r], bins=50, alpha=0.7, label='DistilBERT')
plt.hist([r['mistral_conf'] for r in results if 'mistral_conf' in r and r['mistral_conf'] > 0], bins=50, alpha=0.7, label='Mistral')
plt.xlabel('Confiance')
plt.ylabel('Fréquence')
plt.title('Distribution des Confiances des Modèles')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist([r['final_conf'] for r in results], bins=50, alpha=0.7, color='green')
plt.xlabel('Confiance Finale')
plt.ylabel('Fréquence')
plt.title('Distribution de la Confiance de Fusion')

plt.tight_layout()
plt.show()

# ----------------------------
# 8️⃣ ANALYSE DES ERREURS
# ----------------------------

print("\n🔹 Étape 8: Analyse des erreurs...")

# Identifier les erreurs de la fusion intelligente
errors_df = results_df[results_df['final_pred'] != results_df['true_label']]

print(f"\n❌ NOMBRE D'ERREURS: {len(errors_df)} sur {len(results_df)} ({len(errors_df)/len(results_df)*100:.2f}%)")

if len(errors_df) > 0:
    print(f"\n🔍 ANALYSE DES {len(errors_df)} ERREURS:")
    
    # Erreurs par méthode
    error_methods = errors_df['methode'].value_counts()
    print("\nErreurs par méthode:")
    for method, count in error_methods.items():
        print(f"  {method}: {count}")
    
    # Erreurs par classe originale
    error_classes = errors_df['original_class'].value_counts()
    print("\nErreurs par classe originale:")
    for class_name, count in error_classes.items():
        print(f"  {class_name}: {count}")
    
    # Afficher quelques exemples d'erreurs
    print(f"\n👀 EXEMPLES D'ERREURS (5 premiers):")
    for i, (idx, row) in enumerate(errors_df.head(5).iterrows()):
        print(f"\n📝 Erreur {i+1}:")
        print(f"   Tweet: {row['text'][:100]}...")
        print(f"   🏷️  Classe originale: {row['original_class']}")
        print(f"   ✅ Vrai label: {'Sarcastique' if row['true_label'] == 1 else 'Regular'}")
        print(f"   🤖 Prédiction: {'Sarcastique' if row['final_pred'] == 1 else 'Regular'}")
        print(f"   🔧 Méthode: {row['methode']}")
        print(f"   🎯 Confiance: {row['final_conf']:.3f}")

# ----------------------------
# 9️⃣ SAUVEGARDE DES RÉSULTATS
# ----------------------------

print("\n🔹 Étape 9: Sauvegarde des résultats complets...")

# Sauvegarder les résultats détaillés
results_df.to_csv('resultats_complets_fusion_sarcasme.csv', index=False)

# Sauvegarder le résumé
with open('resume_resultats_complets.txt', 'w') as f:
    f.write("RÉSULTATS COMPLETS - DÉTECTION DE SARCASME\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset testé: {len(test_complete)} tweets\n")
    f.write(f"Distribution: {test_complete['label'].value_counts().to_dict()}\n\n")
    
    f.write("PERFORMANCE DES MODÈLES:\n")
    for model_result in comparison_complete:
        f.write(f"\n{model_result['Model']}:\n")
        f.write(f"  Accuracy: {model_result['Accuracy']}\n")
        f.write(f"  Precision: {model_result['Precision']}\n")
        f.write(f"  Recall: {model_result['Recall']}\n")
        f.write(f"  F1-Score: {model_result['F1-Score']}\n")
    
    f.write(f"\nRÉPARTITION DES MÉTHODES:\n")
    for method, count in method_counts.items():
        f.write(f"  {method}: {count}\n")
    
    if len(errors_df) > 0:
        f.write(f"\nANALYSE DES ERREURS:\n")
        f.write(f"  Total erreurs: {len(errors_df)}\n")
        f.write(f"  Taux d'erreur: {len(errors_df)/len(results_df)*100:.2f}%\n")

print("✅ Résultats sauvegardés dans 'resultats_complets_fusion_sarcasme.csv' et 'resume_resultats_complets.txt'")

# ----------------------------
# 🔟 RAPPORT FINAL
# ----------------------------

print("\n" + "="*100)
print("🏆 RAPPORT FINAL - TEST COMPLET SUR 8,119 TWEETS")
print("="*100)

best_method = max(comparison_complete, key=lambda x: x['F1-Score'])
print(f"🎯 MEILLEURE MÉTHODE: {best_method['Model']}")
print(f"   F1-Score: {best_method['F1-Score']}, Accuracy: {best_method['Accuracy']}")
print(f"   Precision: {best_method['Precision']}, Recall: {best_method['Recall']}")

print(f"\n📈 STATISTIQUES GLOBALES:")
print(f"   • Total tweets testés: {len(test_complete)}")
print(f"   • Taux de succès global: {(len(results_df) - len(errors_df))/len(results_df)*100:.2f}%")
print(f"   • Méthode la plus utilisée: {method_counts.index[0]} ({method_counts.iloc[0]} fois)")

print(f"\n🎉 TEST COMPLET TERMINÉ AVEC SUCCÈS!")
print("="*100)









🔹 Étape 6: Analyse comparative complète...

====================================================================================================
📊 COMPARAISON COMPLÈTE SUR TOUT LE DATASET - 8,119 TWEETS
====================================================================================================
                Model  Accuracy  Precision  Recall  F1-Score
           DistilBERT    1.0000      1.000  1.0000    1.0000
Mistral-7B (Few-Shot)    0.9167      0.917  0.9808    0.9478
     Fusion Parallèle    1.0000      1.000  1.0000    1.0000
  Fusion Séquentielle    1.0000      1.000  1.0000    1.0000
  Fusion Intelligente    1.0000      1.000  1.0000    1.0000



https://www.kaggle.com/code/ahmedyk/notebook1914533164/edit 