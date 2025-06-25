import cv2
from fall_detection import FallDetector
from violence_detection import ViolenceDetector

class SurveillanceSystem:
    def __init__(self):
        self.fall_detector = FallDetector()
        self.violence_detector = ViolenceDetector()        
        self.cap = cv2.VideoCapture(0) # Use 0 for webcam or replace with video file path
        self.running = False
        
    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            fall_frame, fall_status = self.fall_detector.detect(frame.copy())
            violence_frame, violence_status = self.violence_detector.detect(frame.copy())
            
            cv2.imshow('Fall Detection', fall_frame)
            cv2.imshow('Violence Detection', violence_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.running = False
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SurveillanceSystem()
    system.run()