from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self, model_path=None):
        if model_path is None:
             # root/behaviourlogic/analyzers/object_detector.py -> root/models/yolov8n.pt
             base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             model_path = os.path.join(base_dir, "models", "yolov8n.pt")
        
        print(f"ObjectDetector: Loading YOLO from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"ObjectDetector: Error loading YOLO: {e}")
            # Try downloading/loading default if local file fails? 
            # YOLO usually auto-downloads if you just pass "yolov8n.pt" without path, 
            # but providing a path ensures we use the project one.
            self.model = YOLO("yolov8n.pt") 

    def detect_persons(self, frame):
        # Returns list of [x1, y1, x2, y2]
        results = self.model(frame, verbose=False)
        person_boxes = []
        
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0: # 0 is person in COCO
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    person_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return person_boxes
