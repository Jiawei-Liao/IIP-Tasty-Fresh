from ultralytics import YOLO
import torch

class YOLOv8PersonDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov8n.pt").to(self.device)
        self.classes_to_detect = [0]

    def predict(self, image):
        results = self.model.predict(image, conf=0.5, device=self.device, classes=self.classes_to_detect)
        return results

yolov8_person_detector = YOLOv8PersonDetector()