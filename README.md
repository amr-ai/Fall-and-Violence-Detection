# Surveillance System with Fall and Violence Detection

## Overview
This project implements a real-time surveillance system that detects falls and violent behavior using computer vision and deep learning. The system consists of three main components:
1. Fall detection using YOLOv8 object detection
2. Violence detection using a pre-trained CNN model
3. A main surveillance system that combines both detectors

## Files Structure

```
surveillance-system/
├── fall_detection.py       # Fall detection implementation
├── violence_detection.py    # Violence detection implementation
├── main.py                 # Main surveillance system
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Class Documentation

### FallDetector (`fall_detection.py`)
Detects falling persons using YOLOv8 object detection.

**Methods:**
- `__init__(model_path='yolov8n.pt', classes_path='classes.txt')`: Initializes the detector with YOLO model
- `_load_classes(path)`: Loads class names from file
- `detect(frame)`: Processes a frame and returns annotated frame + detection status

### ViolenceDetector (`violence_detection.py`)
Detects violent behavior using a pre-trained CNN model.

**Methods:**
- `__init__(model_path='model.h5')`: Loads the violence detection model
- `preprocess_frame(frame)`: Prepares frame for model input
- `detect(frame)`: Processes a frame and returns annotated frame + detection status

### SurveillanceSystem (`main.py`)
Main system that combines both detectors.

**Methods:**
- `__init__()`: Initializes detectors and video capture
- `run()`: Main loop for processing video feed

## Requirements
Create a `requirements.txt` file with the following content:

```
opencv-python==4.9.0.80
cvzone==1.5.6
ultralytics==8.0.196
numpy==1.26.4
tensorflow==2.15.0
```

## Installation
1. Clone the repository
2. Create and activate a virtual environment
3. Install requirements: `pip install -r requirements.txt`
4. Download the following model files:
   - YOLOv8 model (default will be downloaded automatically)
   - Violence detection model (`model.h5`)
   - Class names file (`classes.txt`)

## Usage
Run the main system:
```bash
python main.py
```

Controls:
- Press 'q' to quit the application

## Notes
- For webcam use, set `cv2.VideoCapture(0)`
- For video file processing, replace with file path
- Adjust confidence thresholds in the detector classes as needed
- The system displays two separate windows for fall and violence detection
