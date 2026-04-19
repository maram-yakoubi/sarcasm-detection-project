# -*- coding: utf-8 -*-
"""Sarcasm Detection - VERSION COMPLÈTE AVEC TOUS LES IMPORTS"""

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
    pipeline  # ✅ IMPORT AJOUTÉ ICI
)

# ----------------------------
# CONFIG OPTIMISÉE POUR GPU
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
print(f"🔹 Device: {device}")

# ----------------------------
# 1) CHARGEMENT DATASETS
# ----------------------------
def load_orabi_dataset(file_path):
    file_path = Path(file_path)
    print(f"\n📥 Chargement de {file_path.name} ...")
    df = pd.read_csv(file_path)
    print(f"   ✅ CSV chargé: {len(df)} lignes")
    if 'class' not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'class'")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df['label'] = df['class'].map({'notsarc': 0, 'sarc': 1})
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""].reset_index(drop=True)
    print(f"   ✅ Après nettoyage: {len(df)} échantillons")
    print(f"   📊 Distribution: {df['class'].value_counts().to_dict()}")
    return df[['text', 'label', 'class']]

# Chemins des datasets
dataset_paths = {
    "RQ": "/kaggle/input/sarcasm-corpus-v2oraby-et-al/RQ-sarc-notsarc.csv",
    "GEN": "/kaggle/input/sarcasm-corpus-v2oraby-et-al/GEN-sarc-notsarc.csv", 
    "HYP": "/kaggle/input/sarcasm-corpus-v2oraby-et-al/HYP-sarc-notsarc.csv"
}

print("🚀 DÉBUT DU CHARGEMENT DES DATASETS...")
datasets = {}
for name, path in dataset_paths.items():
    if os.path.exists(path):
        datasets[name] = load_orabi_dataset(path)
    else:
        print(f"❌ Fichier non trouvé: {path} (skipped)")

print(f"\n🎯 CHARGEMENT TERMINÉ: {len(datasets)} datasets disponibles")

# ----------------------------
# 2) Dataset PyTorch OPTIMISÉ
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
# 3) Callback avec monitoring mémoire
# ----------------------------
class MemoryMonitoringCallback(TrainerCallback):
    def __init__(self, dataset_name=""):
        self.dataset_name = dataset_name

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step % 50 == 0:
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"   💾 Step {state.global_step} - GPU: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

# ----------------------------
# 4) Métriques
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
# 5) Entraînement DistilBERT OPTIMISÉ
# ----------------------------
def train_distilbert_on_dataset(train_df, val_df, dataset_name, model_name="distilbert-base-uncased"):
    print("\n" + "="*60)
    print(f"🎯 ENTRAÎNEMENT DISTILBERT - {dataset_name}")
    print("="*60)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("   🔄 Chargement tokenizer & modèle...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    
    per_device_batch_size = 8
    
    print("   ✅ Modèle & tokenizer chargés")

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

    print(f"   🚀 Lancement de l'entraînement...")
    
    try:
        train_result = trainer.train()
        print("   ✅ Entraînement terminé.")
        
        eval_results = trainer.evaluate()
        print(f"   📊 Résultats finaux - F1: {eval_results.get('eval_f1', 0):.4f}")
        
        return model, tokenizer
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("   ❌ ERREUR: Mémoire GPU insuffisante!")
            return None, None
        else:
            raise e

# ----------------------------
# 6) MISTRAL OPTIMISÉ - VERSION CORRIGÉE
# ----------------------------
class MistralManager:
    def __init__(self):
        self.pipe = None
        self.is_loaded = False
    
    def initialize(self):
        if self.is_loaded:
            return self.pipe
            
        print("\n🔧 Initialisation Mistral...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # ✅ CORRECTION: pipeline est maintenant importé
            self.pipe = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.3",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16,
                max_new_tokens=10,  # Réduit pour plus de rapidité
                do_sample=False,
                temperature=0.0
            )
            self.is_loaded = True
            print("   ✅ Mistral chargé avec succès")
            return self.pipe
        except Exception as e:
            print(f"   ❌ Échec chargement Mistral: {e}")
            return None
    
    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("   🧹 Mistral déchargé de la mémoire GPU")

mistral_manager = MistralManager()

def predict_with_mistral_few_shot(text, mistral_pipe):
    if mistral_pipe is None:
        return 0, 0.5
        
    prompt = f"""<s>[INST] Ce texte est-il sarcastique? Réponds seulement par OUI ou NON.
Texte: "{text[:150]}"
Réponse: [/INST]"""
    
    try:
        outputs = mistral_pipe(
            prompt, 
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=mistral_pipe.tokenizer.eos_token_id,
        )
        
        response = outputs[0]['generated_text'].split("Réponse:")[-1].strip().upper()
        
        if "OUI" in response:
            return 1, 0.9
        elif "NON" in response:
            return 0, 0.9
        else:
            return 0, 0.5
            
    except Exception as e:
        print(f"   ⚠️ Erreur Mistral: {e}")
        return 0, 0.5

# ----------------------------
# 7) TOUTES LES FONCTIONS DE FUSION
# ----------------------------

def fusion_parallele(distil_probs, mistral_pred, mistral_confidence):
    """Fusion parallèle simple - moyenne des probabilités"""
    if mistral_pred == 1:
        mistral_probs = np.array([1 - mistral_confidence, mistral_confidence])
    else:
        mistral_probs = np.array([mistral_confidence, 1 - mistral_confidence])
    fused_probs = (distil_probs + mistral_probs) / 2.0
    return int(np.argmax(fused_probs)), float(np.max(fused_probs))

def analyser_regles_linguistiques(text):
    """Analyse basée sur des motifs linguistiques de sarcasme"""
    text_lower = text.lower()
    indicateurs_forts = [
        "yet again", "of course", "sure,", "obviously", "what a surprise",
        "big surprise", "shocked by", "who would have thought", "another",
        "you're kidding", "are you serious", "oh great", "perfect timing",
        "just what i needed", "so much fun", "how convenient"
    ]
    if any(kw in text_lower for kw in indicateurs_forts):
        return 1
    return 0

def fusion_sequentielle_amelioree(distil_probs, distil_conf, mistral_pred, mistral_conf, text):
    """
    FUSION SÉQUENTIELLE COMPLÈTE
    """
    # 1. Vérifier les règles linguistiques
    regles_pred = analyser_regles_linguistiques(text)
    
    # 2. Si règles détectent du sarcasme ET Mistral confirme
    if regles_pred == 1 and mistral_pred == 1:
        return 1, max(distil_conf, mistral_conf), "Règles + Mistral"
    
    # 3. DistilBERT très confiant
    if distil_conf > 0.95:
        return int(np.argmax(distil_probs)), float(distil_conf), "DistilBERT (très confiant)"
    
    # 4. Mistral très confiant
    if mistral_conf > 0.9:
        return int(mistral_pred), float(mistral_conf), "Mistral (très confiant)"
    
    # 5. Règles linguistiques + accord partiel
    if regles_pred == 1 and mistral_conf > 0.7:
        return 1, max(distil_conf, mistral_conf), "Règles + Mistral (partiel)"
    
    # 6. Les deux modèles sont d'accord
    if np.argmax(distil_probs) == mistral_pred:
        confidence_combined = (distil_conf + mistral_conf) / 2.0
        return int(mistral_pred), float(confidence_combined), "Accord des modèles"
    
    # 7. Désaccord → Priorité à DistilBERT
    return int(np.argmax(distil_probs)), float(distil_conf * 0.8), "DistilBERT (désaccord)"

def fusion_intelligente_amelioree(text, distilbert_model, distilbert_tokenizer, mistral_pipe):
    """
    FUSION INTELLIGENTE COMPLÈTE
    """
    # Inférence DistilBERT
    inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(distilbert_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        distil_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        distil_pred = int(np.argmax(distil_probs))
        distil_conf = float(np.max(distil_probs))
    
    # Inférence Mistral
    mistral_pred, mistral_conf = predict_with_mistral_few_shot(text, mistral_pipe)
    
    # APPLIQUER TOUTES LES FUSIONS
    pred_parallele, conf_parallele = fusion_parallele(distil_probs, mistral_pred, mistral_conf)
    pred_sequentielle, conf_sequentielle, methode_seq = fusion_sequentielle_amelioree(
        distil_probs, distil_conf, mistral_pred, mistral_conf, text
    )
    
    # Décision finale intelligente
    if conf_sequentielle > conf_parallele + 0.15:
        final_pred, final_conf, methode = pred_sequentielle, conf_sequentielle, f"Séquentielle ({methode_seq})"
    elif conf_parallele > conf_sequentielle + 0.15:
        final_pred, final_conf, methode = pred_parallele, conf_parallele, "Parallèle"
    else:
        if pred_parallele == pred_sequentielle:
            final_pred = pred_parallele
            final_conf = (conf_parallele + conf_sequentielle) / 2.0
            methode = f"Accord fusions ({methode_seq})"
        else:
            final_pred, final_conf, methode = pred_sequentielle, conf_sequentielle, f"Séquentielle ({methode_seq})"
    
    return {
        'final_pred': int(final_pred),
        'final_conf': float(final_conf),
        'methode': methode,
        'distil_pred': distil_pred,
        'distil_conf': distil_conf,
        'mistral_pred': mistral_pred,
        'mistral_conf': mistral_conf,
        'parallele_pred': pred_parallele,
        'parallele_conf': conf_parallele,
        'sequentielle_pred': pred_sequentielle,
        'sequentielle_conf': conf_sequentielle
    }

# ----------------------------
# 8) PIPELINE D'ÉVALUATION COMPLÈTE
# ----------------------------
results_summary = {}

for dataset_name, full_df in datasets.items():
    print("\n" + "="*80)
    print(f"🚀 PIPELINE - {dataset_name}")
    print("="*80)

    # Split des données
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    print(f"   🏋️ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Entraînement DistilBERT
    distil_model, distil_tokenizer = train_distilbert_on_dataset(train_df, val_df, dataset_name)
    
    if distil_model is None:
        print("   ⏭️ Passage au dataset suivant...")
        continue

    # Évaluation COMPLÈTE avec toutes les méthodes
    y_true = test_df['label'].tolist()
    methods_preds = {
        "DistilBERT": [],
        "Mistral Few-Shot": [],
        "Fusion Parallèle": [],
        "Fusion Séquentielle": [], 
        "Fusion Intelligente": []
    }

    print(f"\n📊 Évaluation COMPLÈTE sur {len(test_df)} échantillons...")
    
    # Charger Mistral pour l'évaluation
    mistral_pipe = mistral_manager.initialize()
    
    if mistral_pipe is None:
        print("   ⚠️ Évaluation sans Mistral (utilisation de fallbacks)")
    
    pbar = tqdm(total=len(test_df), desc=f"Évaluation {dataset_name}")
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        text = row['text']
        true_label = int(row['label'])

        # Prédiction DistilBERT
        inputs = distil_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(distil_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = distil_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_distil = int(np.argmax(probs))
            conf_distil = float(np.max(probs))

        # Prédiction Mistral (avec fallback si non disponible)
        if mistral_pipe is not None:
            pred_mistral, conf_mistral = predict_with_mistral_few_shot(text, mistral_pipe)
        else:
            pred_mistral, conf_mistral = 0, 0.5  # Fallback

        # TOUTES LES FUSIONS
        pred_parallele, conf_parallele = fusion_parallele(probs, pred_mistral, conf_mistral)
        pred_sequentielle, conf_sequentielle, methode_seq = fusion_sequentielle_amelioree(
            probs, conf_distil, pred_mistral, conf_mistral, text
        )
        res_fusion = fusion_intelligente_amelioree(text, distil_model, distil_tokenizer, mistral_pipe)

        # Stocker TOUTES les prédictions
        methods_preds["DistilBERT"].append(pred_distil)
        methods_preds["Mistral Few-Shot"].append(pred_mistral)
        methods_preds["Fusion Parallèle"].append(pred_parallele)
        methods_preds["Fusion Séquentielle"].append(pred_sequentielle)
        methods_preds["Fusion Intelligente"].append(res_fusion['final_pred'])

        # Mettre à jour la progression
        if (i + 1) % 50 == 0:
            postfix = {}
            for name in methods_preds.keys():
                if len(methods_preds[name]) > 0:
                    acc = accuracy_score(y_true[:i+1], methods_preds[name])
                    postfix[name[:8]] = f'{acc:.3f}'
            pbar.set_postfix(postfix)
        
        pbar.update(1)
    
    pbar.close()
    
    # Libérer la mémoire
    mistral_manager.unload()
    del distil_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calcul des métriques finales
    print(f"\n📊 RÉSULTATS FINAUX ({dataset_name}):")
    print(f"{'Méthode':25s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s}")
    print("-" * 60)
    
    dataset_results = {}
    for name, preds in methods_preds.items():
        acc = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        print(f"{name:25s} {acc:8.3f} {prec:10.3f} {rec:8.3f} {f1:6.3f}")
        dataset_results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    results_summary[dataset_name] = dataset_results

# ----------------------------
# 9) RAPPORT FINAL COMPLET
# ----------------------------
print("\n" + "="*80)
print("🎉 RAPPORT FINAL - TOUTES LES MÉTHODES DE FUSION")
print("="*80)

if results_summary:
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    methods_list = ["DistilBERT", "Mistral Few-Shot", "Fusion Parallèle", "Fusion Séquentielle", "Fusion Intelligente"]

    for metric in metrics:
        print(f"\n📊 COMPARAISON - {metric.upper()}:")
        header = f"{'Dataset':10s}" + "".join([f"{m:>18s}" for m in methods_list])
        print(header)
        print("-" * (10 + 18 * len(methods_list)))
        for ds in results_summary.keys():
            row = f"{ds:10s}"
            for method in methods_list:
                val = results_summary[ds].get(method, {}).get(metric, 0)
                row += f"{val:18.3f}"
            print(row)

    # Trouver la meilleure méthode globale
    print(f"\n🏆 MEILLEURE MÉTHODE GLOBALE:")
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
    print("❌ Aucun résultat à afficher")

print("\n✅ PIPELINE COMPLÈTE TERMINÉE! 🎉")


🎉 RAPPORT FINAL - TOUTES LES MÉTHODES DE FUSION
================================================================================

📊 COMPARAISON - ACCURACY:
Dataset           DistilBERT  Mistral Few-Shot  Fusion ParallèleFusion SéquentielleFusion Intelligente
----------------------------------------------------------------------------------------------------
RQ                     0.748             0.587             0.587             0.710             0.721

📊 COMPARAISON - PRECISION:
Dataset           DistilBERT  Mistral Few-Shot  Fusion ParallèleFusion SéquentielleFusion Intelligente
----------------------------------------------------------------------------------------------------
RQ                     0.804             0.551             0.551             0.710             0.690

📊 COMPARAISON - RECALL:
Dataset           DistilBERT  Mistral Few-Shot  Fusion ParallèleFusion SéquentielleFusion Intelligente
----------------------------------------------------------------------------------------------------
RQ                     0.653             0.929             0.929             0.706             0.800

📊 COMPARAISON - F1:
Dataset           DistilBERT  Mistral Few-Shot  Fusion ParallèleFusion SéquentielleFusion Intelligente
----------------------------------------------------------------------------------------------------
RQ                     0.721             0.691             0.691             0.708             0.741




📊 RÉSULTATS FINAUX (RQ):
Méthode                   Accuracy  Precision   Recall     F1
------------------------------------------------------------
DistilBERT                   0.748      0.804    0.653  0.721
Mistral Few-Shot             0.587      0.551    0.929  0.691
Fusion Parallèle             0.587      0.551    0.929  0.691
Fusion Séquentielle          0.710      0.710    0.706  0.708
Fusion Intelligente          0.721      0.690    0.800  0.741






https://www.kaggle.com/code/maramyakoubi/notebook0ccccff2fe/edit





# -*- coding: utf-8 -*-
"""Sarcasm Detection with Dual Fusion - GEN Dataset ONLY (Same Architecture)"""

# Installations nécessaires
!pip install transformers datasets evaluate accelerate tqdm --quiet

import os
import torch
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, pipeline
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Device: {device}")

# ----------------------------
# 1️⃣ CHARGEMENT ET NETTOYAGE - GEN SEULEMENT
# ----------------------------

print("🔹 Étape 1: Chargement dataset GEN seulement...")

# Charger les données GEN seulement
df = pd.read_csv('/kaggle/input/sarcasm-corpus-v2oraby-et-al/GEN-sarc-notsarc.csv')

# Renommer les colonnes (identique)
df = df.rename(columns={"tweets": "text"})
df['label'] = df['class'].map({'notsarc': 0, 'sarc': 1})

# Nettoyage des données (identique)
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'] != ""]

print(f"📊 GEN Dataset size: {len(df)}")
print(f"📊 Distribution GEN: {df['label'].value_counts()}")

# ----------------------------
# 2️⃣ SPLIT DES DONNÉES (identique)
# ----------------------------

print("🔹 Étape 2: Split train/val...")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}")
print(f"📊 Distribution train: {train_df['label'].value_counts()}")

# ----------------------------
# 3️⃣ FINE-TUNING DISTILBERT - OPTIMISÉ POUR MÉMOIRE
# ----------------------------

print("🔹 Étape 3: Fine-tuning de DistilBERT sur GEN...")

# Configuration DistilBERT (identique)
distilbert_model_name = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    distilbert_model_name, 
    num_labels=2
).to(device)

# Dataset class (identique)
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

# Préparation des datasets (identique)
train_data = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), distilbert_tokenizer)
val_data = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), distilbert_tokenizer)

# Métriques (identique)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Arguments d'entraînement OPTIMISÉS pour mémoire
training_args = TrainingArguments(
    output_dir="./distilbert-gen-only",
    num_train_epochs=2,  # Réduit
    per_device_train_batch_size=8,  # Réduit
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=30,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    gradient_accumulation_steps=2,  # Ajouté pour mémoire
    dataloader_pin_memory=False,   # Optimisation mémoire
)

# Trainer (identique)
trainer = Trainer(
    model=distilbert_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Entraînement
print("🔹 Début de l'entraînement DistilBERT sur GEN...")
trainer.train()

# Évaluation sur validation
distilbert_results = trainer.evaluate(val_data)
print("📊 Résultats DistilBERT sur GEN:", distilbert_results)

# ----------------------------
# 4️⃣ FONCTIONS DE FUSION INTELLIGENTES - IDENTIQUES
# ----------------------------

print("🔹 Étape 4: Configuration des fusions intelligentes...")

# Charger Mistral (identique)
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
    """Prédiction avec Mistral en few-shot learning - IDENTIQUE"""
    if mistral_pipe is None:
        return 0, 0.5
    
    # Prompt few-shot avec exemples concrets de tweets
    prompt = f"""<s>[INST] Analyse ces tweets et détermine s'ils contiennent du sarcasme.
Réponds UNIQUEMENT par "SARCASTIQUE" ou "NORMAL".

Exemples:
Tweet: "Oh great, another Monday. Just what I needed." → SARCASTIQUE
Tweet: "I love waiting in line for 2 hours. So much fun!" → SARCASTIQUE  
Tweet: "The weather is beautiful today." → NORMAL
Tweet: "My phone battery died at 10%. How convenient." → SARCASTIQUE
Tweet: "I just finished all my work on time." → NORMAL
Tweet: "Another perfect day with no surprises at all." → SARCASTIQUE
Tweet: "The package arrived exactly when they said it would." → NORMAL

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
        if "SARCASTIQUE" in cleaned_response and "NORMAL" not in cleaned_response:
            return 1, 0.9
        elif "NORMAL" in cleaned_response:
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

# Les fonctions de fusion RESTENT IDENTIQUES
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
    
    # Indicateurs de sarcasme adaptés pour les tweets
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
    
    # Utiliser la fonction few-shot
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
# 5️⃣ ÉVALUATION SUR LE TEST SET GEN
# ----------------------------

print("🔹 Étape 5: Évaluation des fusions intelligentes sur GEN...")

# Utiliser le set de validation comme test (puisque nous n'avons pas de test séparé)
test_sample = val_df
y_true = test_sample['label'].tolist()
results = []

print(f"🔹 Test sur {len(test_sample)} tweets GEN...")
print(f"📊 Distribution des labels: {pd.Series(y_true).value_counts()}")

for i, text in enumerate(tqdm(test_sample['text'].tolist(), desc="Prédictions GEN")):
    try:
        result = fusion_intelligente(text, distilbert_model, distilbert_tokenizer)
        result['true_label'] = y_true[i]
        result['text'] = text
        results.append(result)
        
    except Exception as e:
        print(f"Erreur: {e}")
        inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(distilbert_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            distil_pred = torch.argmax(outputs.logits).cpu().item()
        
        results.append({
            'final_pred': distil_pred,
            'final_conf': 0.8,
            'methode': "Fallback (DistilBERT)",
            'true_label': y_true[i],
            'text': text
        })

# ----------------------------
# 6️⃣ ANALYSE COMPARATIVE
# ----------------------------

# Extraire les prédictions
y_pred_distil = [r['distil_pred'] for r in results if 'distil_pred' in r]
y_pred_mistral = [r['mistral_pred'] for r in results if 'mistral_pred' in r]
y_pred_fusion = [r['final_pred'] for r in results]
y_pred_parallele = [r['parallele_pred'] for r in results if 'parallele_pred' in r]
y_pred_sequentielle = [r['sequentielle_pred'] for r in results if 'sequentielle_pred' in r]

# Métriques
def calculate_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }

# Comparaison
comparison = []
comparison.append(calculate_metrics(y_true, y_pred_distil, "DistilBERT"))
comparison.append(calculate_metrics(y_true, y_pred_mistral, "Mistral-7B (Few-Shot)"))
comparison.append(calculate_metrics(y_true, y_pred_parallele, "Fusion Parallèle"))
comparison.append(calculate_metrics(y_true, y_pred_sequentielle, "Fusion Séquentielle"))
comparison.append(calculate_metrics(y_true, y_pred_fusion, "Fusion Intelligente"))

# Affichage
print("\n" + "="*80)
print("📊 COMPARAISON COMPLÈTE DES MÉTHODES - GEN DATASET")
print("="*80)
print(pd.DataFrame(comparison).to_string(index=False))

# Analyse des méthodes
method_counts = pd.Series([r['methode'] for r in results]).value_counts()
print(f"\n🔧 RÉPARTITION DES MÉTHODES ({len(results)} prédictions):")
for method, count in method_counts.items():
    print(f"  {method}: {count}")

# ----------------------------
# 7️⃣ RÉSULTATS DÉTAILLÉS
# ----------------------------

print("\n" + "="*80)
print("🏆 RÉSULTATS - DÉTECTION DE SARCASME - GEN DATASET")
print("="*80)

best_method = max(comparison, key=lambda x: x['F1-Score'])
print(f"🎯 MEILLEURE MÉTHODE: {best_method['Model']}")
print(f"   F1-Score: {best_method['F1-Score']}, Accuracy: {best_method['Accuracy']}")

# Exemples détaillés
print(f"\n👀 EXEMPLES DÉTAILLÉS (3 premiers):")
for i, result in enumerate(results[:3]):
    true_label_desc = "Sarcastique" if result['true_label'] == 1 else "Normal"
    
    print(f"\n📝 Exemple {i+1}:")
    print(f"   Tweet: {result['text'][:80]}...")
    print(f"   ✅ Vrai label: {true_label_desc}")
    print(f"   🤖 DistilBERT: {'Sarcastique' if result.get('distil_pred', -1) == 1 else 'Normal'}")
    print(f"   🌟 Mistral: {'Sarcastique' if result.get('mistral_pred', -1) == 1 else 'Normal'}")
    print(f"   ⚡ Fusion: {'Sarcastique' if result['final_pred'] == 1 else 'Normal'}")
    print(f"   🔧 Méthode: {result['methode']}")
    print(f"   🎯 Correct: {result['final_pred'] == result['true_label']}")

print("\n" + "="*80)
print("🎉 PROCESSUS GEN TERMINÉ AVEC SUCCÈS!")
print("="*80)



📊 COMPARAISON COMPLÈTE DES MÉTHODES - GEN DATASET
================================================================================
                Model  Accuracy  Precision  Recall  F1-Score
           DistilBERT    0.7699     0.7924  0.7316    0.7608
Mistral-7B (Few-Shot)    0.6511     0.6019  0.8926    0.7190
     Fusion Parallèle    0.7232     0.6622  0.9110    0.7669
  Fusion Séquentielle    0.7554     0.7542  0.7577    0.7559
  Fusion Intelligente    0.7623     0.7436  0.8006    0.7710





# -*- coding: utf-8 -*-
"""Sarcasm Detection with Dual Fusion - HYP Dataset ONLY (Same Architecture)"""

# Installations nécessaires
!pip install transformers datasets evaluate accelerate tqdm --quiet

import os
import torch
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, pipeline
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Device: {device}")

# ----------------------------
# 1️⃣ CHARGEMENT ET NETTOYAGE - HYP SEULEMENT
# ----------------------------

print("🔹 Étape 1: Chargement dataset HYP seulement...")

# Charger les données HYP seulement
df = pd.read_csv('/kaggle/input/sarcasm-corpus-v2oraby-et-al/HYP-sarc-notsarc.csv')

# Renommer les colonnes (identique)
df = df.rename(columns={"tweets": "text"})
df['label'] = df['class'].map({'notsarc': 0, 'sarc': 1})

# Nettoyage des données (identique)
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'] != ""]

print(f"📊 HYP Dataset size: {len(df)}")
print(f"📊 Distribution HYP: {df['label'].value_counts()}")

# ----------------------------
# 2️⃣ SPLIT DES DONNÉES (identique)
# ----------------------------

print("🔹 Étape 2: Split train/val...")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}")
print(f"📊 Distribution train: {train_df['label'].value_counts()}")

# ----------------------------
# 3️⃣ FINE-TUNING DISTILBERT - ULTRA OPTIMISÉ POUR MÉMOIRE
# ----------------------------

print("🔹 Étape 3: Fine-tuning de DistilBERT sur HYP...")

# Configuration DistilBERT (identique)
distilbert_model_name = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    distilbert_model_name, 
    num_labels=2
).to(device)

# Dataset class (identique)
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

# Préparation des datasets (identique)
train_data = SarcasmDataset(train_df['text'].tolist(), train_df['label'].tolist(), distilbert_tokenizer)
val_data = SarcasmDataset(val_df['text'].tolist(), val_df['label'].tolist(), distilbert_tokenizer)

# Métriques (identique)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Arguments d'entraînement ULTRA OPTIMISÉS pour mémoire
training_args = TrainingArguments(
    output_dir="./distilbert-hyp-only",
    num_train_epochs=2,
    per_device_train_batch_size=4,  # ULTRA RÉDUIT
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    gradient_accumulation_steps=4,  # HAUTE ACCUMULATION
    dataloader_pin_memory=False,
)

# Trainer (identique)
trainer = Trainer(
    model=distilbert_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Entraînement
print("🔹 Début de l'entraînement DistilBERT sur HYP...")
trainer.train()

# Évaluation sur validation
distilbert_results = trainer.evaluate(val_data)
print("📊 Résultats DistilBERT sur HYP:", distilbert_results)

# ----------------------------
# 4️⃣ FONCTIONS DE FUSION INTELLIGENTES - IDENTIQUES
# ----------------------------

print("🔹 Étape 4: Configuration des fusions intelligentes...")

# Charger Mistral (identique)
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
    """Prédiction avec Mistral en few-shot learning - IDENTIQUE"""
    if mistral_pipe is None:
        return 0, 0.5
    
    # Prompt few-shot avec exemples concrets de tweets
    prompt = f"""<s>[INST] Analyse ces tweets et détermine s'ils contiennent du sarcasme.
Réponds UNIQUEMENT par "SARCASTIQUE" ou "NORMAL".

Exemples:
Tweet: "Oh great, another Monday. Just what I needed." → SARCASTIQUE
Tweet: "I love waiting in line for 2 hours. So much fun!" → SARCASTIQUE  
Tweet: "The weather is beautiful today." → NORMAL
Tweet: "My phone battery died at 10%. How convenient." → SARCASTIQUE
Tweet: "I just finished all my work on time." → NORMAL
Tweet: "Another perfect day with no surprises at all." → SARCASTIQUE
Tweet: "The package arrived exactly when they said it would." → NORMAL

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
        if "SARCASTIQUE" in cleaned_response and "NORMAL" not in cleaned_response:
            return 1, 0.9
        elif "NORMAL" in cleaned_response:
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

# Les fonctions de fusion RESTENT IDENTIQUES
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
    
    # Indicateurs de sarcasme adaptés pour les tweets
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
    
    # Utiliser la fonction few-shot
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
# 5️⃣ ÉVALUATION SUR LE TEST SET HYP
# ----------------------------

print("🔹 Étape 5: Évaluation des fusions intelligentes sur HYP...")

# Utiliser le set de validation comme test
test_sample = val_df
y_true = test_sample['label'].tolist()
results = []

print(f"🔹 Test sur {len(test_sample)} tweets HYP...")
print(f"📊 Distribution des labels: {pd.Series(y_true).value_counts()}")

for i, text in enumerate(tqdm(test_sample['text'].tolist(), desc="Prédictions HYP")):
    try:
        result = fusion_intelligente(text, distilbert_model, distilbert_tokenizer)
        result['true_label'] = y_true[i]
        result['text'] = text
        results.append(result)
        
    except Exception as e:
        print(f"Erreur: {e}")
        inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(distilbert_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            distil_pred = torch.argmax(outputs.logits).cpu().item()
        
        results.append({
            'final_pred': distil_pred,
            'final_conf': 0.8,
            'methode': "Fallback (DistilBERT)",
            'true_label': y_true[i],
            'text': text
        })

# ----------------------------
# 6️⃣ ANALYSE COMPARATIVE
# ----------------------------

# Extraire les prédictions
y_pred_distil = [r['distil_pred'] for r in results if 'distil_pred' in r]
y_pred_mistral = [r['mistral_pred'] for r in results if 'mistral_pred' in r]
y_pred_fusion = [r['final_pred'] for r in results]
y_pred_parallele = [r['parallele_pred'] for r in results if 'parallele_pred' in r]
y_pred_sequentielle = [r['sequentielle_pred'] for r in results if 'sequentielle_pred' in r]

# Métriques
def calculate_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }

# Comparaison
comparison = []
comparison.append(calculate_metrics(y_true, y_pred_distil, "DistilBERT"))
comparison.append(calculate_metrics(y_true, y_pred_mistral, "Mistral-7B (Few-Shot)"))
comparison.append(calculate_metrics(y_true, y_pred_parallele, "Fusion Parallèle"))
comparison.append(calculate_metrics(y_true, y_pred_sequentielle, "Fusion Séquentielle"))
comparison.append(calculate_metrics(y_true, y_pred_fusion, "Fusion Intelligente"))

# Affichage
print("\n" + "="*80)
print("📊 COMPARAISON COMPLÈTE DES MÉTHODES - HYP DATASET")
print("="*80)
print(pd.DataFrame(comparison).to_string(index=False))

# Analyse des méthodes
method_counts = pd.Series([r['methode'] for r in results]).value_counts()
print(f"\n🔧 RÉPARTITION DES MÉTHODES ({len(results)} prédictions):")
for method, count in method_counts.items():
    print(f"  {method}: {count}")

# ----------------------------
# 7️⃣ RÉSULTATS DÉTAILLÉS
# ----------------------------

print("\n" + "="*80)
print("🏆 RÉSULTATS - DÉTECTION DE SARCASME - HYP DATASET")
print("="*80)

best_method = max(comparison, key=lambda x: x['F1-Score'])
print(f"🎯 MEILLEURE MÉTHODE: {best_method['Model']}")
print(f"   F1-Score: {best_method['F1-Score']}, Accuracy: {best_method['Accuracy']}")

# Exemples détaillés
print(f"\n👀 EXEMPLES DÉTAILLÉS (3 premiers):")
for i, result in enumerate(results[:3]):
    true_label_desc = "Sarcastique" if result['true_label'] == 1 else "Normal"
    
    print(f"\n📝 Exemple {i+1}:")
    print(f"   Tweet: {result['text'][:80]}...")
    print(f"   ✅ Vrai label: {true_label_desc}")
    print(f"   🤖 DistilBERT: {'Sarcastique' if result.get('distil_pred', -1) == 1 else 'Normal'}")
    print(f"   🌟 Mistral: {'Sarcastique' if result.get('mistral_pred', -1) == 1 else 'Normal'}")
    print(f"   ⚡ Fusion: {'Sarcastique' if result['final_pred'] == 1 else 'Normal'}")
    print(f"   🔧 Méthode: {result['methode']}")
    print(f"   🎯 Correct: {result['final_pred'] == result['true_label']}")

print("\n" + "="*80)
print("🎉 PROCESSUS HYP TERMINÉ AVEC SUCCÈS!")
print("="*80)






📊 COMPARAISON COMPLÈTE DES MÉTHODES - HYP DATASET
================================================================================
                Model  Accuracy  Precision  Recall  F1-Score
           DistilBERT    0.6609     0.6178  0.8362    0.7106
Mistral-7B (Few-Shot)    0.6009     0.5578  0.9569    0.7048
     Fusion Parallèle    0.6009     0.5567  0.9741    0.7085
  Fusion Séquentielle    0.6438     0.5965  0.8793    0.7108
  Fusion Intelligente    0.6009     0.5567  0.9741    0.7085


  https://www.kaggle.com/code/maramyakoubi/notebooke79555eb9b/edit



