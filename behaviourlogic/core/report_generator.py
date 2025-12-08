from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
import torch
from PIL import Image
import os

class ReportGenerator:
    def __init__(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"ReportGenerator: Loading models on {self.device}...")
            
            # Load CLIP (Vision)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Load GPT-2 (Text)
            self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            self.models_loaded = True
            print("ReportGenerator: Models loaded successfully.")
        except Exception as e:
            print(f"ReportGenerator: Error loading models: {e}")
            self.models_loaded = False

    def generate_detailed_description(self, image_input):
        if not self.models_loaded:
            return "Error: AI Models not loaded."

        try:
            # Handle both file path and PIL Image/Numpy Array
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return f"Error: File not found at {image_input}"
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else: # Assyume numpy array (OpenCV)
                image = Image.fromarray(image_input)

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Extract features (This is a simplified pipeline, normally we'd project CLIP features to GPT2 space
            # but since we don't have a trained mapping network here, we will use a text prompt + unconditional generation 
            # or a very basic heuristic. 
            # NOTE: The original code was using a placeholder approach (text prompt -> GPT2). 
            # It wasn't actually feeding image features into GPT2 (which requires an Image Captioning model like BLIP, not just CLIP+GPT2 raw).
            # To fix this "cleanly" as requested, I should switch to a proper Image Captioning pipeline or stick to the original placeholder logic if user insisted on CLIP+GPT2.
            # Given "fill components" and "work best", I will implement a better prompt using CLIP to classify scene first, then prompt GPT2.
            
            # Improved Logic:
            # 1. Use CLIP to detect key elements (Violence, Harassment, Calm, Night, Crowd)
            # 2. Feed these keywords to GPT2 to generate a sentence.
            
            labels = ["violence", "fighting", "harassment", "safe", "crowd", "alone", "night", "day", "running", "falling"]
            inputs = self.clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                
            # Get top 3 labels
            values, indices = probs[0].topk(3)
            detected_concepts = [labels[idx] for idx in indices]
            
            # Generate Report
            prompt = f"Surveillance Detection Report: observed {', '.join(detected_concepts)}."
            
            # Refine with GPT2
            input_ids = self.gpt2_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            output = self.gpt2_model.generate(
                input_ids,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            description = self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
            return description

        except Exception as e:
            print(f"ReportGenerator: Error processing image: {e}")
            return "Error generating report."
