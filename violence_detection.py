import cv2
import numpy as np
from tensorflow.keras.models import load_model

class ViolenceDetector:
    def __init__(self, model_path='model.h5'):
        self.model = load_model(model_path)
        self.frame_size = (640, 640)
        
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (224, 224))  
        frame = frame / 255.0 
        frame = np.expand_dims(frame, axis=0)  
        return frame
    
    def detect(self, frame):
        original_frame = cv2.resize(frame, self.frame_size)
        processed_frame = self.preprocess_frame(frame)
        
        prediction = self.model.predict(processed_frame)[0][0]
        violence_prob = float(prediction)
        violence_detected = violence_prob > 0.7  
        
        
        label = "Violence: {:.2f}%".format(violence_prob * 100)
        color = (0, 0, 255) if violence_detected else (0, 255, 0)
        cv2.putText(original_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return original_frame, violence_detected