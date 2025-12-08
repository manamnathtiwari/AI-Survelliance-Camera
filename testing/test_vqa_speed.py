import time
import cv2
import os
import sys
# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaviourlogic.physical_assault.physical_assault import ViolenceDetector

def test_speed():
    print("Initializing ViolenceDetector (1000+ Questions)...")
    detector = ViolenceDetector()
    
    # Create a dummy image or load one
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    video_path = os.path.join(assets_dir, "sample_people.mp4")
    
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("\nStarting Inference on a Single Frame...")
    start_time = time.time()
    
    status, reason = detector.detect_violence(frame)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nInference Complete!")
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Result: {status}")
    print(f"Reason: {reason}")
    
    print(f"\nEstimated FPS: {1/duration:.4f}")

if __name__ == "__main__":
    test_speed()
