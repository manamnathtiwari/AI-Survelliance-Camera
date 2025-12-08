import cv2
import time
import mediapipe as mp
import os
from mtcnn import MTCNN 

# Import Custom Modules
from behaviourlogic.core.tracker import CentroidTracker
from behaviourlogic.core.alert_system import AlertSystem
from behaviourlogic.analyzers.object_detector import ObjectDetector
from behaviourlogic.analyzers.face_analyzer import FaceAnalyzer
from behaviourlogic.analyzers.pose_analyzer import PoseAnalyzer
from behaviourlogic.physical_assault.physical_assault import ViolenceDetector
from behaviourlogic.harrassmentdetection.harrassmentdetection import HarassmentDetector
from behaviourlogic.AccidentDetection.accidentdetection import AccidentDetector

def main():
    print("Initializing Mirage Surveillance System...")
    
    # Initialize Components
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    tracker = CentroidTracker()
    object_detector = ObjectDetector() # YOLOv8
    face_analyzer = FaceAnalyzer() # Gender + Emotion
    pose_analyzer = PoseAnalyzer() # Action/Pose
    violence_detector = ViolenceDetector() # LSTM Model
    harassment_detector = HarassmentDetector() # Logic
    accident_detector = AccidentDetector() # Fall Logic
    alert_system = AlertSystem() 
    
    # MTCNN for face crop (Better alignment than just raw crop if needed, or simple crop from person)
    # The original code used MTCNN inside the loop. YOLO provides person box. MTCNN finds face in person box.
    mtcnn_detector = MTCNN() 
    mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

    skip_frame = 5
    frame_count = 0
    
    print("System Active. Press 'q' to exit.")

    while True:
        status, frame = webcam.read()
        if not status:
            print("Failed to read frame.")
            break

        frame_count += 1
        if frame_count % skip_frame != 0:
            cv2.imshow("Mirage Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 1. Detect Persons
        person_boxes = object_detector.detect_persons(frame)
        
        # 2. Track Objects
        objects = tracker.update(person_boxes)
        
        frame_data = {'persons': []} # Collect data for high level logic
        
        for objectID, centroid in objects.items():
            # Find closest box for this centroid (approximation)
            # Tracker returns centroids, we need to map back to boxes for cropping features.
            # Simplified: re-match or store box in tracker. For now, let's find closest box.
            # Actually, `objects` in this tracker implementation only stores centroids.
            # We need the box to crop. Let's assume the box index matches closely or just re-match.
            # A better tracker tracks boxes. 
            # Workaround: Find box containing centroid.
            
            x_c, y_c = centroid
            best_box = None
            for box in person_boxes:
                bx1, by1, bx2, by2 = box
                if bx1 <= x_c <= bx2 and by1 <= y_c <= by2:
                    best_box = box
                    break
            
            if best_box is None: continue
            
            x1, y1, x2, y2 = best_box
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0: continue

            # Feature Extraction
            
            # A. Violence
            # Note: ViolenceDetector now works on FULL FRAME usually for VQA scene understanding
            # But here we are calling it per person? 
            # If we switch to VQA, we should call it once per frame, not per person.
            # However, for minimal diff, let's keep it here but pass full frame? 
            # NO, VQA is scene level. Let's move it out of the person loop.
            # For now, just fix the call structure to unpack tuple.
            violence_label, violence_reason = violence_detector.detect_violence(person_img)
            
            # B. Face Analysis (Gender, Emotion)
            # Detect face in person crop
            faces = mtcnn_detector.detect_faces(person_img)
            gender = "Unknown"
            emotion = "Unknown"
            face_box = None
            
            if faces:
                f = faces[0]
                fx, fy, fw, fh = f['box']
                face_img = person_img[fy:fy+fh, fx:fx+fw]
                gender, emotion = face_analyzer.analyze_face(face_img)
                # Draw Face Box
                cv2.rectangle(frame, (x1+fx, y1+fy), (x1+fx+fw, y1+fy+fh), (255, 255, 0), 1)

            # C. Pose Analysis
            # Use Holistic or Pose on full person image
            pose_label = "Unknown"
            # Convert to RGB for mediapipe
            results = mp_holistic.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                pose_label = pose_analyzer.analyze_pose(results.pose_landmarks)
                # Draw Pose (Optional, heavy)
                mp.solutions.drawing_utils.draw_landmarks(frame[y1:y2, x1:x2], results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Store Data
            p_data = {
                'id': objectID,
                'bbox': [x1, y1, x2, y2],
                'gender': gender,
                'emotion': emotion,
                'pose': pose_label,
                'violence': violence_label
            }
            frame_data['persons'].append(p_data)
            
            # Annotate
            label = f"ID:{objectID} {gender} {emotion} {pose_label}"
            color = (0, 255, 0)
            if violence_label == "Violence":
                label += " [VIOLENCE]"
                color = (0, 0, 255)
                alert_system.send_alert(frame, f"Violence Detected! ID: {objectID}. Reason: {violence_reason}")

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 3. High Level Logic (Harassment, Accident, Chasing)
        
        # Chasing Detection
        chasing_events = tracker.detect_chasing() # Returns list of (id1, id2)
        for c1, c2 in chasing_events:
             cv2.putText(frame, "CHASING BEHAVIOUR", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
             alert_system.send_alert(frame, f"Chasing Detected between ID {c1} and ID {c2}")

        # Harassment
        harassment_alerts = harassment_detector.detect_harassment(frame_data, None)
        for alert in harassment_alerts:
            cv2.putText(frame, "HARASSMENT ALERT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            alert_system.send_alert(frame, alert)

        # Accident (Fall)
        accident_alerts = accident_detector.detect_accident(frame_data['persons'])
        for alert in accident_alerts:
             cv2.putText(frame, "ACCIDENT ALERT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
             alert_system.send_alert(frame, alert)
        
        # Display Stats
        males = sum(1 for p in frame_data['persons'] if p['gender'] == 'male')
        females = sum(1 for p in frame_data['persons'] if p['gender'] == 'female')
        cv2.putText(frame, f"M: {males} F: {females} Total: {len(frame_data['persons'])}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Mirage Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
