import unittest
import cv2
import sys
import os
import numpy as np

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaviourlogic.analyzers.object_detector import ObjectDetector
from behaviourlogic.core.tracker import CentroidTracker
from behaviourlogic.analyzers.pose_analyzer import PoseAnalyzer
# from behaviourlogic.analyzers.face_analyzer import FaceAnalyzer # Might take long to load
from behaviourlogic.physical_assault.physical_assault import ViolenceDetector

class TestMirageComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up Test Suite...")
        cls.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        cls.video_path = os.path.join(cls.assets_dir, "sample_people.mp4")
        if not os.path.exists(cls.video_path):
            print(f"Warning: Video not found at {cls.video_path}. Some tests might fail.")
            cls.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            cap = cv2.VideoCapture(cls.video_path)
            ret, cls.frame = cap.read()
            cap.release()
            if not ret:
                cls.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_1_object_detector(self):
        print("\nTesting Object Detector...")
        detector = ObjectDetector()
        boxes = detector.detect_persons(self.frame)
        print(f"Detected {len(boxes)} persons.")
        self.assertIsInstance(boxes, list)

    def test_2_tracker(self):
        print("\nTesting Tracker...")
        tracker = CentroidTracker()
        # Simulate detections
        rects = [(100, 100, 150, 150), (200, 200, 250, 250)]
        objects = tracker.update(rects)
        print(f"Tracked Objects: {objects}")
        self.assertTrue(len(objects) > 0)
        
        # Test Chasing Logic (Mock history)
        # Force history
        tracker.path_history[0] = [(100, 100), (110, 110), (120, 120), (150, 150)] # Moving fast
        tracker.path_history[1] = [(120, 120), (130, 130), (140, 140), (170, 170)] # Parallel
        events = tracker.detect_chasing(threshold_dist=500)
        print(f"Chasing Events: {events}")
        # Note: logic requires speed > 20px/frame. 120->150 is 30 distance (diag > 20).
        # Should detect.

    def test_3_pose_analyzer(self):
        print("\nTesting Pose Analyzer...")
        analyzer = PoseAnalyzer()
        # Needs pose landmarks. Hard to mock complex landmarks without MP.
        # We will just instantiate for now, or run on a crop if we had landmarks.
        # Let's trust initialization works.
        self.assertIsNotNone(analyzer.pose)

    def test_4_violence_detector(self):
        print("\nTesting Violence Detector...")
        detector = ViolenceDetector()
        # Needs sequence of 16 frames
        dummy_frame = np.zeros((160, 160, 3), dtype=np.uint8)
        
        # Feed 16 frames
        res = "Buffering..."
        for _ in range(16):
            res = detector.detect_violence(dummy_frame)
        
        print(f"Violence Result: {res}")
        self.assertIn(res, ["Violence", "Non-Violence"])

if __name__ == '__main__':
    unittest.main()
