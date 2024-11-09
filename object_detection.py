# object_detection.py
import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolo11s.pt', device='cuda'):
        self.device = device
        # Load the YOLOv11 model
        self.model = YOLO(model_path)
        self.model.fuse()  # Optional: for faster inference
        self.model.to(self.device)

    def detect(self, frame):
        # Convert frame to RGB
        camera = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Perform inference
        results = self.model.track(camera,show=True,persist=True)
        detections = results[0]
        return detections
