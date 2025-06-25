import cv2
import cvzone
import math
from ultralytics import YOLO

class FallDetector:
    def __init__(self, model_path='yolov8n.pt', classes_path='classes.txt'):
        self.model = YOLO(model_path)
        self.classnames = self._load_classes(classes_path)
        self.frame_size = (640, 640)
        
    def _load_classes(self, path):
        with open(path, 'r') as f:
            return f.read().splitlines()
    
    def detect(self, frame):
        frame = cv2.resize(frame, self.frame_size)
        results = self.model(frame, verbose=False)[0]
        
        fall_detected = False
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.classnames[class_id]
            conf = math.ceil(confidence * 100)
            
            if conf > 80 and class_name == 'person':
                height = y2 - y1
                width = x2 - x1
                threshold = height - width
                
                # رسم المستطيل والنص
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_name}', [x1 + 8, y1 - 12], thickness=2, scale=2)
                
                # كشف السقوط
                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                    fall_detected = True
        
        return frame, fall_detected