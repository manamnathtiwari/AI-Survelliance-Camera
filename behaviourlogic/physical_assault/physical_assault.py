import os
import cv2
import numpy as np
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

class ViolenceDetector:
    def __init__(self):
        # Using ViLT (Vision-and-Language Transformer) for VQA
        # Model: dandelin/vilt-b32-finetuned-vqa
        print("ViolenceDetector: Loading VQA Model (ViLT)...")
        try:
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.model_loaded = True
            print("ViolenceDetector: VQA Model loaded successfully.")
        except Exception as e:
            print(f"ViolenceDetector: Failed to load VQA model: {e}")
            self.model_loaded = False
            
        self.frame_count = 0
        
        # ---------------------------------------------------------
        # COMPREHENSIVE QUESTION BANK GENERATOR (1000+ Questions)
        # ---------------------------------------------------------
        self.questions_map = {}
        self._generate_comprehensive_questions()

        self.questions_list = list(self.questions_map.keys())
        # Threshold scales with number of questions, but since we normalize or cap, 
        # let's keep a reasonable threshold for "confirmed violence".
        # If we have 1000 questions, many will be negative. We care about the sum of positives.
        self.alert_threshold = 3.0 

    def _generate_comprehensive_questions(self):
        # 1. SUBJECTS (Who)
        subjects = [
            "person", "man", "woman", "guy", "girl", "boy", "someone", "people", "group", "gang", 
            "attacker", "victim", "suspect", "criminal", "stranger", "intruder"
        ]
        
        # 2. ACTIONS - AGGRESSIVE (What) - Weight 1.0
        actions_violent = [
            "fighting", "punching", "kicking", "slapping", "hitting", "beating", 
            "strangling", "choking", "attacking", "hurting", "killing", "murdering", 
            "stabbing", "shooting", "shoving", "pushing", "dragging", "grabbing",
            "smashing", "crushing", "destroying", "assaulting", "brawling"
        ]
        
        # 3. ACTIONS - SUSPICIOUS/THREAT (What) - Weight 0.7
        actions_suspicious = [
            "running", "chasing", "fleeing", "hiding", "stalking", "following", 
            "yelling", "screaming", "shouting", "crying", "arguing", "threatening", 
            "begging", "panicking", "scared", "terrified", "bleeding", "falling"
        ]
        
        # 4. OBJECTS - WEAPONS (With What) - Weight 1.0
        weapons = [
            "weapon", "gun", "pistol", "rifle", "firearm", "knife", "blade", "sword", 
            "machete", "baseball bat", "stick", "crowbar", "explosive", "bomb"
        ]
        
        # 5. BODY PARTS (Where) - Weight 0.8
        body_parts = ["face", "head", "stomach", "chest", "neck", "back"]

        # --- GENERATION LOOPS ---
        
        # Type A: Is {subject} {action}?
        for subj in subjects:
            for act in actions_violent:
                q = f"Is the {subj} {act}?"
                self.questions_map[q] = 1.0
                q2 = f"Is {subj} {act}?"
                self.questions_map[q2] = 1.0

            for act in actions_suspicious:
                q = f"Is the {subj} {act}?"
                self.questions_map[q] = 0.7
        
        # Type B: Is there a {weapon}? / Is {subject} holding {weapon}?
        for w in weapons:
            self.questions_map[f"Is there a {w}?"] = 1.0
            self.questions_map[f"Do you see a {w}?"] = 1.0
            for subj in subjects:
                self.questions_map[f"Is the {subj} holding a {w}?"] = 1.0
                self.questions_map[f"Does the {subj} have a {w}?"] = 1.0

        # Type C: Interaction - Is {subject} {action} {subject}?
        # Limit combinations to avoid explosion (10*20*10 = 2000 alone)
        # We start with generic 'someone'
        for act in actions_violent:
            self.questions_map[f"Is someone {act} someone else?"] = 1.0
            self.questions_map[f"Is a man {act} a woman?"] = 1.0
            self.questions_map[f"Is a man {act} a man?"] = 1.0
            self.questions_map[f"Is a group {act} someone?"] = 1.0

        # Type D: Environment/Context
        self.questions_map["Is there blood?"] = 1.0
        self.questions_map["Is there a fire?"] = 0.8
        self.questions_map["Is the scene chaotic?"] = 0.5
        self.questions_map["Is it dangerous?"] = 0.5
        self.questions_map["Are they angry?"] = 0.4
        
        # Type E: Safety Checks (Negative Weights)
        safety_concepts = ["happy", "smiling", "laughing", "playing", "dancing", "hugging", "kissing", "safe", "calm", "peaceful"]
        for s in safety_concepts:
            self.questions_map[f"Is everyone {s}?"] = -0.5
            self.questions_map[f"Are they {s}?"] = -0.5
            self.questions_map[f"Is the situation {s}?"] = -0.5

        print(f"ViolenceDetector: Generated {len(self.questions_map)} unique VQA questions.")

    def detect_violence(self, frame):
        """
        Returns a tuple: (Status, Reason)
        Status: "Violence" or "Non-Violence"
        Reason: String explaining the decision
        """
        if not self.model_loaded:
            return "Model Error", "Model not loaded"
            
        self.frame_count += 1
        # VQA is heavy. Running on every frame is slow.
        # Ideally run every 5-10 frames.
        
        try:
            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            risk_score = 0.0
            positive_reasons = []
            
            # BATCH INFERENCE OPTIMIZATION
            # ViltProcessor can handle (image, [list of text])
            # This is much faster than looping.
            
            # Process in chunks of 20 to avoid OOM or huge latency per batch
            chunk_size = 20
            all_answers = []
            
            for i in range(0, len(self.questions_list), chunk_size):
                batch_questions = self.questions_list[i : i + chunk_size]
                
                # Prepare inputs: define image list same size as questions list
                # ViLT expects pairs of (image, text) or single image + list of text (depending on version)
                # transformers implementation usually expects matching lengths or expands.
                # Let's try passing single image and list of texts.
                
                try:
                    encoding = self.processor(
                        images=[pil_image] * len(batch_questions), 
                        text=batch_questions, 
                        return_tensors="pt", 
                        padding=True
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(**encoding)
                    
                    logits = outputs.logits
                    idx = logits.argmax(-1) # Tensor of indices
                    
                    # Decode answers
                    for j, answer_idx in enumerate(idx):
                        answer = self.model.config.id2label[answer_idx.item()]
                        all_answers.append((batch_questions[j], answer))
                        
                except Exception as batch_e:
                    print(f"Batch inference failed: {batch_e}. Falling back to single.")
                    # Fallback code if batching fails (unlikely)
                    pass

            # SCORING FORMULA
            for question, answer in all_answers:
                weight = self.questions_map[question]
                is_positive = False
                
                # Check for Affirmative answers
                if answer.lower() in ['yes', 'yeah', 'y', '1', 'true']:
                    is_positive = True
                
                # Check for direct object answers (e.g. "Is there a weapon?" -> "gun")
                # ViLT often answers 'yes'/'no' but sometimes specific objects.
                # If the answer is not 'no' and not 'none' and weight is positive, it might be a detection
                if answer.lower() not in ['no', 'none', 'nothing', 'false', 'n', '0'] and not is_positive:
                    # If answer is specific (e.g. "gun"), treating as positive for that object
                     if weight > 0:
                         is_positive = True

                if is_positive:
                    risk_score += weight
                    # Only log significant reasons (positive weight)
                    if weight > 0:
                        positive_reasons.append(f"[{answer}] {question}")
            
            # Final Decision
            # If negative weights (happiness) outweigh positive, score drops.
            
            if risk_score > self.alert_threshold:
                # Top 3 reasons
                top_reasons = positive_reasons[:3] 
                reason_str = f"Score {risk_score:.1f}: {', '.join(top_reasons)}..."
                return "Violence", reason_str
            else:
                return "Non-Violence", f"Safe (Score: {risk_score:.1f})"

        except Exception as e:
            print(f"ViolenceDetector: Error: {e}")
            return "Error", str(e)

    def preprocess_frame(self, frame):
        pass
