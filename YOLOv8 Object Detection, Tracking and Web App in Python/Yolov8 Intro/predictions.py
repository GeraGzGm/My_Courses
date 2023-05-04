from ultralytics import YOLO

# Initialize model name
model = YOLO("yolov8n.pt")

# Call predict method
# model.predict(source="data/image1.jpg", save=True, save_txt=True)
model.export(format='onnx')
