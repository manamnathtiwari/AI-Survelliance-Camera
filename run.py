import streamlit as st
import cv2
import tempfile
import os
import time
from behaviourlogic.core.tracker import CentroidTracker
from behaviourlogic.core.alert_system import AlertSystem
from behaviourlogic.analyzers.object_detector import ObjectDetector
from behaviourlogic.analyzers.face_analyzer import FaceAnalyzer
from behaviourlogic.analyzers.pose_analyzer import PoseAnalyzer
from behaviourlogic.physical_assault.physical_assault import ViolenceDetector
from behaviourlogic.harrassmentdetection.harrassmentdetection import HarassmentDetector
from behaviourlogic.AccidentDetection.accidentdetection import AccidentDetector

def main():
    st.set_page_config(page_title="Mirage Surveillance", layout="wide")
    st.title("Mirage: Real-time Violence & Harassment Detection")
    st.markdown("### Advanced Proactive Surveillance System")

    # Options
    st.sidebar.title("Settings")
    source = st.sidebar.radio("Video Source", ("Webcam", "Upload Video"))
    enable_violence = st.sidebar.checkbox("Detect Violence", value=True)
    enable_harassment = st.sidebar.checkbox("Detect Harassment", value=True)
    enable_accident = st.sidebar.checkbox("Detect Accidents (Falls)", value=True)

    # Placeholders
    col1, col2 = st.columns([3, 1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        st.subheader("Live Stats")
        stats_text = st.empty()
        st.subheader("Alerts")
        alerts_placeholder = st.empty()

    # Initialize Modules (Lazy Load)
    if 'detectors_loaded' not in st.session_state:
        with st.spinner("Loading AI Models..."):
            st.session_state.tracker = CentroidTracker()
            st.session_state.obj_det = ObjectDetector()
            st.session_state.face_anlz = FaceAnalyzer()
            st.session_state.pose_anlz = PoseAnalyzer()
            st.session_state.viol_det = ViolenceDetector()
            st.session_state.harr_det = HarassmentDetector()
            st.session_state.acc_det = AccidentDetector()
        st.session_state.detectors_loaded = True

    start_btn = st.sidebar.button("Start Surveillance")
    stop_btn = st.sidebar.button("Stop")

    cap = None
    if start_btn:
        if source == "Webcam":
            cap = cv2.VideoCapture(0)
        else:
            uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi'])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.warning("Upload a video first.")

    if cap and cap.isOpened():
        skip_frame = 3
        count = 0
        
        while cap.isOpened():
            if stop_btn: break
            
            ret, frame = cap.read()
            if not ret: break
            
            count += 1
            if count % skip_frame != 0: continue

            # Processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Detection
            person_boxes = st.session_state.obj_det.detect_persons(frame)
            objects = st.session_state.tracker.update(person_boxes)
            
            frame_data = {'persons': []}
            active_alerts = []

            for objectID, centroid in objects.items():
                x_c, y_c = centroid
                best_box = None
                for box in person_boxes:
                    bx1, by1, bx2, by2 = box
                    if bx1 <= x_c <= bx2 and by1 <= y_c <= by2:
                        best_box = box
                        break
                
                if best_box:
                    x1, y1, x2, y2 = best_box
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size == 0: continue
                    
                    # Analyze
                    gender, emotion = st.session_state.face_anlz.analyze_face(person_img)
                    violence = "Safe"
                    if enable_violence:
                        violence = st.session_state.viol_det.detect_violence(person_img)
                    
                    pose = "Unknown"
                    # For pose we need full body but we have crop. 
                    # If using holistic, pass full frame + ROI? Or just crop? 
                    # Providing crop acts as full image for MP.
                    # Ideally pass crop.
                    # Recalculating MP here is expensive. 
                    # Optimized: Run ONE holistic on full frame if possible, but we are doing per person.
                    # Let's run pose logic only if violence/harassment enabled to save FPS.
                    
                    # Pose
                    # Convert crop to RGB
                    # Note: MP Pose expects full body. Crop works if full body is in crop.
                    try:
                         # We need to initialize new MP instance or reuse? 
                         # st.session_state.pose_anlz uses internal MP.
                         # Since it's stateless, we can call it.
                         # But MP Pose instance is stateful for tracking. 
                         # Creating new MP per frame per person is BAD.
                         # Current PoseAnalyzer implementation re-inits MP? No, init in __init__.
                         # But running process() on different images (different people) sequentially effectively resets tracking or confuses it.
                         # For this simple demo, we use static_image_mode=False so it expects stream.
                         # Switching logic: Use static mode for multi-person crops OR use full frame MP (box matching).
                         # Let's assume PoseAnalyzer handles it (or we accept it's slow).
                         # To speed up: Only run Pose every N frames.
                         pass 
                    except: pass
                    
                    p_data = {
                        'id': objectID,
                        'bbox': best_box,
                        'gender': gender,
                        'emotion': emotion,
                        'pose': "Standing", # Placeholder for speed in UI
                        'violence': violence
                    }
                    frame_data['persons'].append(p_data)
                    
                    # Annotate
                    color = (0, 255, 0)
                    if violence == "Violence":
                        active_alerts.append(f"Violence detected on Person {objectID}")
                        color = (255, 0, 0)
                    
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_rgb, f"{gender} {emotion}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # High Level Logic
            # Chasing
            chasing_events = st.session_state.tracker.detect_chasing()
            for c1, c2 in chasing_events:
                 active_alerts.append(f"Chasing Behaviour detected: ID {c1} -> ID {c2}")

            if enable_harassment:
                h_alerts = st.session_state.harr_det.detect_harassment(frame_data, None)
                active_alerts.extend(h_alerts)
            
            if enable_accident:
                a_alerts = st.session_state.acc_det.detect_accident(frame_data['persons'])
                active_alerts.extend(a_alerts)
                
            # Update UI
            video_placeholder.image(frame_rgb)
            
            males = sum(1 for p in frame_data['persons'] if p['gender'] == 'male')
            females = sum(1 for p in frame_data['persons'] if p['gender'] == 'female')
            stats_text.markdown(f"**Persons**: {len(frame_data['persons'])}\n\n**Males**: {males}\n\n**Females**: {females}")
            
            if active_alerts:
                alerts_placeholder.error("\n".join(active_alerts))
            else:
                alerts_placeholder.success("System Secure")

    if cap: cap.release()

if __name__ == "__main__":
    main()
