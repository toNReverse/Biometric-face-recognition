from ultralytics import YOLO

model = YOLO("yolo11n.pt")  
model.train(data="YOLO11-DATASET\data.yaml", epochs=100, imgsz=640)