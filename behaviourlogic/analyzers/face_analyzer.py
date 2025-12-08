import cv2
import numpy as np
import os
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

class FaceAnalyzer:
    def __init__(self):
        # Gender Classifier (HF Pipeline)
        try:
            print("FaceAnalyzer: Loading Gender Model...")
            self.gender_classifier = pipeline("image-classification", model="rizvandwiki/gender-classification")
        except Exception as e:
            print(f"FaceAnalyzer: Gender Model Error: {e}")
            self.gender_classifier = None

        # Emotion Detector (Custom Keras)
        try:
            print("FaceAnalyzer: Loading Emotion Model...")
            # Automatically find the model in models folder
            # Current file is in behaviourlogic/analyzers/
            # Path: root/behaviourlogic/analyzers/face_analyzer.py -> need root/models/
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "emotion_detection_model_50epochs.h5")
            
            if not os.path.exists(model_path):
                 print(f"FaceAnalyzer: Warning - Emotion model not found at {model_path}")
            
            self.emotion_model = load_model(model_path)
            self.emotion_labels = ['Fear', 'Happy', 'Neutral', 'Sad']
        except Exception as e:
            print(f"FaceAnalyzer: Emotion Model Error: {e}")
            self.emotion_model = None

    def analyze_face(self, face_img):
        """
        Returns (Gender, Emotion) tuple
        """
        if face_img.shape[0] < 10 or face_img.shape[1] < 10:
            return "Unknown", "Neutral"

        gender = self._detect_gender(face_img)
        emotion = self._detect_emotion(face_img)
        return gender, emotion

    def _detect_gender(self, face_img):
        if not self.gender_classifier:
            return "Unknown"
        try:
            rgb_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            results = self.gender_classifier(images=pil_image)
            # Result is list of dicts [{'label': 'male', 'score': 0.9}, ...]
            # We want the top one
            top_result = results[0]
            label = top_result['label'].lower()
            return label # 'male' or 'female'
        except:
            return "Unknown"

    def _detect_emotion(self, face_img):
        if not self.emotion_model:
            return "Neutral"
        try:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            roi = face_resized.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) # Batch dim
            
            preds = self.emotion_model.predict(roi, verbose=0)
            idx = np.argmax(preds)
            if idx < len(self.emotion_labels):
                return self.emotion_labels[idx]
            return "Neutral"
        except:
            return "Neutral"
