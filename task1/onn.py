from ultralytics import YOLO
# Load a models
model = YOLO('yolov8n.pt')  # load an official models
# Export the models
model.export(format='onnx', opset=12)
