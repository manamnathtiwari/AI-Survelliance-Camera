import math
import time
from datetime import datetime

class HarassmentDetector:
    def __init__(self):
        self.abnormal_proximity_threshold = 100 # Pixels, adjust based on resolution
        self.night_start = 18
        self.night_end = 6

    def detect_harassment(self, frame_data, context):
        """
        frame_data: dict containing list of persons, genders, emotions, poses
        context: time of day, location (optional)
        """
        alerts = []
        
        persons = frame_data.get('persons', []) # List of dicts: {'id': 0, 'bbox': [x,y,w,h], 'gender': 'male', 'emotion': 'Neutral', 'pose': 'Standing'}
        male_count = sum(1 for p in persons if p['gender'] == 'male')
        female_count = sum(1 for p in persons if p['gender'] == 'female')
        
        females = [p for p in persons if p['gender'] == 'female']
        males = [p for p in persons if p['gender'] == 'male']

        current_hour = datetime.now().hour

        # 1. Check Night Safety (sOS Condition 1)
        if female_count == 1 and male_count == 0:
            if current_hour >= self.night_start or current_hour < self.night_end:
                 alerts.append("Warning: Lone female detected at night.")

        # 2. Check Surrounded (SOS Condition 2)
        if female_count == 1 and male_count >= 3:
            female = females[0]
            if self.is_surrounded(female, males):
                alerts.append("CRITICAL: Female surrounded by multiple males!")

        # 3. Check Harassment (Proximity + Negative Emotion)
        for female in females:
            # Check proximity to any male
            for male in males:
                dist = self.calculate_distance(female['bbox'], male['bbox'])
                if dist < self.abnormal_proximity_threshold:
                    if female['emotion'] in ['Fear', 'Distress', 'Sad'] or male['pose'] in ['Punching', 'Lunging', 'Attack']:
                        alerts.append(f"CRITICAL: Potential Harassment detected (ID {female['id']} & ID {male['id']})")
        
        return alerts

    def is_surrounded(self, female, males, radius=200):
        # Count males within radius
        close_males = 0
        for male in males:
            dist = self.calculate_distance(female['bbox'], male['bbox'])
            if dist < radius:
                close_males += 1
        return close_males >= 3

    def calculate_distance(self, bbox1, bbox2):
        # bbox = [x, y, w, h]
        # or [x1, y1, x2, y2] - Assuming [x1, y1, x2, y2] based on previous code
        c1_x = (bbox1[0] + bbox1[2]) / 2
        c1_y = (bbox1[1] + bbox1[3]) / 2
        
        c2_x = (bbox2[0] + bbox2[2]) / 2
        c2_y = (bbox2[1] + bbox2[3]) / 2
        
        return math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
