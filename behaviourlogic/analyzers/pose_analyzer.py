import cv2
import mediapipe as mp
import math
import numpy as np

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def analyze_pose(self, pose_landmarks):
        if not pose_landmarks:
            return "Unknown"

        # Key Landmarks
        landmarks = pose_landmarks.landmark
        
        # Coordinates (x,y)
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Face landmarks for phone check
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]

        # 1. Defencing / Surrender (Hands above head)
        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            if left_wrist.y < left_eye.y:
                return "Surrender/HandsUp"

        # 2. Punching (Arm extended horizontally)
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        if right_arm_angle > 150: 
            if abs(right_wrist.y - right_shoulder.y) < 0.2:
                return "Punching/Pointing"
        
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        if left_arm_angle > 150:
            if abs(left_wrist.y - left_shoulder.y) < 0.2:
                return "Punching/Pointing"
        
        # 3. Suspicious Photography / Holding Phone
        # Logic: One or both hands near face level, but elbows bent (not punching)
        # Check Right Hand
        if right_wrist.y < right_shoulder.y and right_arm_angle < 100:
            # Wrist near eye level
            if abs(right_wrist.y - right_eye.y) < 0.15:
                # Check intersection with center of face view (simplified)
                if abs(right_wrist.x - nose.x) < 0.15:
                    return "Phone/Photography"
        
        # Check Left Hand
        if left_wrist.y < left_shoulder.y and left_arm_angle < 100:
            if abs(left_wrist.y - left_eye.y) < 0.15:
                if abs(left_wrist.x - nose.x) < 0.15:
                    return "Phone/Photography"

        # 4. Lunging / Wide Stance
        ankle_dist = math.dist([left_ankle.x, left_ankle.y], [right_ankle.x, right_ankle.y])
        shoulder_dist = math.dist([left_shoulder.x, left_shoulder.y], [right_shoulder.x, right_shoulder.y])
        
        if ankle_dist > 2.5 * shoulder_dist:
             return "Lunging/WideStance"

        # 5. Walking
        if ankle_dist > 0.5 * shoulder_dist and ankle_dist < 2.5 * shoulder_dist:
             return "Walking" 
        
        return "Standing"

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
