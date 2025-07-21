from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8x-world.pt")

# Train the model on the COCO8 dataset for 100 epochs
results = model.train(data="SKU-110K.yaml", epochs=20, imgsz=640)