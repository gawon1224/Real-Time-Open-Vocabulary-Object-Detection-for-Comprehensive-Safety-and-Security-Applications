from ultralytics import YOLO
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def initialize():
    # Initialize the YOLO-World model
    model = YOLO("/home/gawon/ultra/runs/detect/train4/weights/best.pt")
    processor = None
    return processor, model

def object_detection(processor, model, image_path, objects):
    model.set_classes(objects)
    results = model.predict(image_path, verbose=False)

    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            label_index = int(box.cls)
            label = objects[label_index]
            score = round(float(box.conf), 2)
            xyxy = box.xyxy[0].tolist()
            box_coords = [round(coord, 2) for coord in xyxy]
            detected_objects.append({"label": label, "score": score, "box": box_coords})
    return detected_objects

if __name__ == "__main__":
    processor, model = initialize()
    image_path = "/home/device03/ultra/KakaoTalk_20241203_202058694.png"
    objects = ["black car", "red car"]

    detection_results = object_detection(processor, model, image_path, objects)
    print(detection_results)