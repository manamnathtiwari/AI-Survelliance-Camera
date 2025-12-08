import cv2
import os
import sys
import numpy as np
from collections import defaultdict

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaviourlogic.analyzers.object_detector import ObjectDetector
from behaviourlogic.core.tracker import CentroidTracker
from behaviourlogic.analyzers.pose_analyzer import PoseAnalyzer
from behaviourlogic.analyzers.face_analyzer import FaceAnalyzer
from behaviourlogic.physical_assault.physical_assault import ViolenceDetector
from behaviourlogic.harrassmentdetection.harrassmentdetection import HarassmentDetector
from behaviourlogic.AccidentDetection.accidentdetection import AccidentDetector

# Mock Alert System to collect alerts
class MockAlertSystem:
    def __init__(self):
        self.alerts = []
    def send_alert(self, frame, message):
        self.alerts.append(message)

def process_video(video_path):
    print(f"\n{'='*50}")
    print(f"Analyzing Video: {os.path.basename(video_path)}")
    print(f"{'='*50}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Initialize Components
    print("Initializing AI Models...")
    try:
        obj_detector = ObjectDetector()
        tracker = CentroidTracker()
        pose_analyzer = PoseAnalyzer()
        face_analyzer = FaceAnalyzer()
        violence_detector = ViolenceDetector()
        harassment_detector = HarassmentDetector()
        accident_detector = AccidentDetector()
        alert_system = MockAlertSystem()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    frame_count = 0
    stats = defaultdict(int)
    detected_events = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 != 0: continue # Process every 10th frame for speed

        # 1. Object Detection
        person_boxes = obj_detector.detect_persons(frame)
        objects = tracker.update(person_boxes)
        
        persons_data = []
        
        # 2. Analysis per person
        for (objectID, centroid) in objects.items():
            # Find closest box for this centroid (approximation)
            # In a real run we map index, but here we just take the box if available
            # This is a simplified binding for testing report
            
            # Crop person for analysis if possible
            # Need strict box mapping. For test summary, let's just analyze detected boxes
            pass

        # We need strict mapping for individual analysis. 
        # Reusing logic from main.py simplified
        
        for box in person_boxes:
            x1, y1, x2, y2 = box
            person_img = frame[y1:y2, x1:x2]
            
            # Attributes
            gender, emotion = face_analyzer.analyze_face(person_img)
            
            # Pose - crop needs to be somewhat centered or use full frame with ROI
            # PoseAnalyzer expects crop or full frame? It takes landmarks.
            # Using full frame with ROI logic is complex here.
            # Let's just run pose on the crop for simplicity in testing
            # (Note: Pose works best on full body usually)
            pose_landmarks = None 
            try:
                pose_results = pose_analyzer.pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                pose_landmarks = pose_results.pose_landmarks
            except: pass
            
            pose = pose_analyzer.analyze_pose(pose_landmarks)

            persons_data.append({
                'id': 0, # Dummy ID for aggregate stats
                'bbox': box,
                'gender': gender,
                'emotion': emotion,
                'pose': pose
            })
            
            # Stats
            stats[gender] += 1
            stats[emotion] += 1
            stats[pose] += 1

        # 3. High Level Event Detection
        
        # Violence
        v_res, v_reason = violence_detector.detect_violence(frame)
        if v_res == "Violence":
            detected_events.add(f"Violence ({v_reason})")
            stats['Violence_Frames'] += 1

        # Harassment
        context = {}
        frame_data = {'persons': persons_data}
        h_alerts = harassment_detector.detect_harassment(frame_data, context)
        for alert in h_alerts:
            detected_events.add(alert)
            alert_system.send_alert(frame, alert)

        # Accident
        a_alerts = accident_detector.detect_accident(persons_data)
        for alert in a_alerts:
            detected_events.add("Accident/Fall")
            alert_system.send_alert(frame, alert)
            
        # Chasing
        chasing = tracker.detect_chasing()
        if chasing:
            detected_events.add("Chasing Behaviour")

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()

    # GENERATE REPORT
    print(f"\n{'-'*20} REPORT: {os.path.basename(video_path)} {'-'*20}")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Detected Events: {', '.join(detected_events) if detected_events else 'None'}")
    print("\nStatistics (Aggregate Counts):")
    for k, v in stats.items():
        if v > 0:
            print(f"  - {k}: {v}")
    
    print("\nAlerts Triggered:")
    unique_alerts = set(alert_system.alerts)
    if unique_alerts:
        for alert in unique_alerts:
            print(f"  [ALERT] {alert}")
    else:
        print("  None")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    videos = [f for f in os.listdir(assets_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    
    if not videos:
        print("No videos found in testing/assets/")
    else:
        for video in videos:
            process_video(os.path.join(assets_dir, video))
