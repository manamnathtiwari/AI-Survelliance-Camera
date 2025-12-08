import cv2
import os
import sys
import numpy as np

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaviourlogic.physical_assault.physical_assault import ViolenceDetector

def verify_video(video_filename):
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    video_path = os.path.join(assets_dir, video_filename)
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        print("Please place a video file there to test.")
        return

    print(f"Processing {video_filename}...")
    cap = cv2.VideoCapture(video_path)
    detector = ViolenceDetector()
    
    frame_count = 0
    violence_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        result = detector.detect_violence(frame)
        
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: {result}")
            
        if result == "Violence":
            violence_count += 1

    cap.release()
    print("-" * 30)
    print(f"Total Frames: {frame_count}")
    print(f"Violence Detected Frames: {violence_count}")
    if violence_count > 0:
        print("RESULT: VIOLENCE DETECTED ✅")
    else:
        print("RESULT: NO VIOLENCE DETECTED ❌ (This is correct for non-violent videos)")

if __name__ == "__main__":
    # Check for violence sample first, then fallback to people sample
    if os.path.exists(os.path.join(os.path.dirname(__file__), "assets", "sample_violence.mp4")):
        verify_video("sample_violence.mp4")
    else:
        print("sample_violence.mp4 not found. Testing on sample_people.mp4 (Control Test)...")
        verify_video("sample_people.mp4")
