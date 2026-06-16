# ifm.py - Intelligent Fusion Module for Sarcasm Detection
# Shared logic for all dataset experiments

import os
import re
import torch
import numpy as np
from transformers import pipeline

class IntelligentFusionMethods:
    """
    Intelligent Fusion Module (IFM) for sarcasm detection.
    Combines DistilBERT, Mistral-7B, and linguistic rules through
    parallel, sequential, and intelligent fusion strategies.
    """
    
    def __init__(self, distilbert_model, distilbert_tokenizer, mistral_pipe=None):
        """
        Initialize IFM with models.
        
        Args:
            distilbert_model: Fine-tuned DistilBERT model
            distilbert_tokenizer: DistilBERT tokenizer
            mistral_pipe: Mistral pipeline (optional)
        """
        self.distilbert_model = distilbert_model
        self.distilbert_tokenizer = distilbert_tokenizer
        self.mistral_pipe = mistral_pipe
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def predict_with_mistral(self, text):
        """
        Predict sarcasm using Mistral-7B with few-shot prompting.
        
        Args:
            text: Input text to analyze
            
        Returns:
            tuple: (prediction, confidence) where prediction is 0 (normal) or 1 (sarcastic)
        """
        if self.mistral_pipe is None:
            return 0, 0.5
        
        # Few-shot prompt with examples
        prompt = f"""<s>[INST] Analyze these tweets and determine if they contain sarcasm.
Reply ONLY with "SARCASTIC" or "NORMAL".

Examples:
Tweet: "Oh great, another Monday. Just what I needed." → SARCASTIC
Tweet: "I love waiting in line for 2 hours. So much fun!" → SARCASTIC  
Tweet: "The weather is beautiful today." → NORMAL
Tweet: "My phone battery died at 10%. How convenient." → SARCASTIC
Tweet: "I just finished all my work on time." → NORMAL
Tweet: "Another perfect day with no surprises at all." → SARCASTIC
Tweet: "The package arrived exactly when they said it would." → NORMAL

Now analyze this tweet:
Tweet: "{text[:150]}"

Answer: [/INST]"""
        
        try:
            outputs = self.mistral_pipe(
                prompt,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.mistral_pipe.tokenizer.eos_token_id,
            )
            
            response = outputs[0]['generated_text'].split("Answer:")[-1].strip().upper()
            
            if "SARCASTIC" in response and "NORMAL" not in response:
                return 1, 0.9
            elif "NORMAL" in response:
                return 0, 0.9
            elif "SARCASTIC" in response:
                return 1, 0.9
            else:
                # Fallback based on keywords
                sarcasm_keywords = ["great", "love", "perfect", "obviously", 
                                   "of course", "sure", "wow", "😂", "😏", "/s"]
                if any(kw in text.lower() for kw in sarcasm_keywords):
                    return 1, 0.7
                return 0, 0.5
                
        except Exception as e:
            print(f"Mistral error: {e}")
            # Fallback based on common sarcasm patterns
            sarcasm_patterns = [
                r"\b(great|love|perfect|obviously|of course|sure|wow)\b.*[!?]",
                r"😂|😏|/s",
                r"\bjust what i needed\b",
                r"\bso much fun\b"
            ]
            if any(re.search(pattern, text.lower()) for pattern in sarcasm_patterns):
                return 1, 0.6
            return 0, 0.5
    
    def get_distilbert_prediction(self, text):
        """
        Get DistilBERT prediction and probabilities.
        
        Args:
            text: Input text to analyze
            
        Returns:
            tuple: (prediction, confidence, probabilities)
        """
        inputs = self.distilbert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        inputs = {k: v.to(self.distilbert_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
        
        return pred, conf, probs
    
    def analyze_linguistic_rules(self, text):
        """
        Analyze text for linguistic sarcasm indicators.
        
        Args:
            text: Input text to analyze
            
        Returns:
            int: 1 if sarcastic indicators found, 0 otherwise
        """
        text_lower = text.lower()
        strong_indicators = [
            "yet again", "of course", "sure,", "obviously", "what a surprise",
            "big surprise", "shocked by", "who would have thought", "another",
            "you're kidding", "are you serious", "oh great", "perfect timing",
            "just what i needed", "so much fun", "how convenient",
            "#sarcasm", "#irony", "#figurative", "😂", "😏", "/s"
        ]
        return 1 if any(kw in text_lower for kw in strong_indicators) else 0
    
    def parallel_fusion(self, distil_probs, mistral_pred, mistral_conf):
        """
        Parallel fusion: average of model probabilities.
        
        Args:
            distil_probs: DistilBERT probability distribution
            mistral_pred: Mistral prediction (0 or 1)
            mistral_conf: Mistral confidence
            
        Returns:
            tuple: (prediction, confidence)
        """
        if mistral_pred == 1:
            mistral_probs = np.array([1 - mistral_conf, mistral_conf])
        else:
            mistral_probs = np.array([mistral_conf, 1 - mistral_conf])
        
        fused_probs = (distil_probs + mistral_probs) / 2.0
        return int(np.argmax(fused_probs)), float(np.max(fused_probs))
    
    def sequential_fusion(self, distil_probs, distil_conf, mistral_pred, mistral_conf, text):
        """
        Sequential fusion: hierarchical decision making.
        
        Args:
            distil_probs: DistilBERT probability distribution
            distil_conf: DistilBERT confidence
            mistral_pred: Mistral prediction (0 or 1)
            mistral_conf: Mistral confidence
            text: Input text for rule analysis
            
        Returns:
            tuple: (prediction, confidence, method_used)
        """
        # 1. Check linguistic rules
        rules_pred = self.analyze_linguistic_rules(text)
        
        # 2. Rules + Mistral agreement
        if rules_pred == 1 and mistral_pred == 1:
            return 1, max(distil_conf, mistral_conf), "Rules + Mistral"
        
        # 3. DistilBERT very confident
        if distil_conf > 0.95:
            return int(np.argmax(distil_probs)), float(distil_conf), "DistilBERT (very confident)"
        
        # 4. Mistral very confident
        if mistral_conf > 0.9:
            return int(mistral_pred), float(mistral_conf), "Mistral (very confident)"
        
        # 5. Rules + partial Mistral agreement
        if rules_pred == 1 and mistral_conf > 0.7:
            return 1, max(distil_conf, mistral_conf), "Rules + Mistral (partial)"
        
        # 6. Model agreement
        if np.argmax(distil_probs) == mistral_pred:
            combined_conf = (distil_conf + mistral_conf) / 2.0
            return int(mistral_pred), float(combined_conf), "Model agreement"
        
        # 7. Disagreement → fallback to DistilBERT
        return int(np.argmax(distil_probs)), float(distil_conf * 0.8), "DistilBERT (default)"
    
    def intelligent_fusion(self, text):
        """
        Complete intelligent fusion with dynamic method selection.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Complete prediction results with all details
        """
        # Get DistilBERT prediction
        distil_pred, distil_conf, distil_probs = self.get_distilbert_prediction(text)
        
        # Get Mistral prediction
        mistral_pred, mistral_conf = self.predict_with_mistral(text)
        
        # Apply both fusion methods
        parallel_pred, parallel_conf = self.parallel_fusion(distil_probs, mistral_pred, mistral_conf)
        sequential_pred, sequential_conf, sequential_method = self.sequential_fusion(
            distil_probs, distil_conf, mistral_pred, mistral_conf, text
        )
        
        # Dynamic selection
        if parallel_conf > sequential_conf + 0.05:
            final_pred, final_conf, method = parallel_pred, parallel_conf, "Parallel Fusion"
        elif sequential_conf > parallel_conf + 0.05:
            final_pred, final_conf, method = sequential_pred, sequential_conf, f"Sequential Fusion ({sequential_method})"
        else:
            if parallel_pred == sequential_pred:
                final_pred = parallel_pred
                final_conf = (parallel_conf + sequential_conf) / 2.0
                method = f"Fusion agreement ({sequential_method})"
            else:
                final_pred, final_conf, method = sequential_pred, sequential_conf, f"Sequential Fusion ({sequential_method})"
        
        return {
            'final_pred': final_pred,
            'final_conf': final_conf,
            'method': method,
            'distil_pred': distil_pred,
            'distil_conf': distil_conf,
            'mistral_pred': mistral_pred,
            'mistral_conf': mistral_conf,
            'parallel_pred': parallel_pred,
            'parallel_conf': parallel_conf,
            'sequential_pred': sequential_pred,
            'sequential_conf': sequential_conf,
            'sequential_method': sequential_method
        }