from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for a smaller model or 'yolov8x.pt' for a larger model

# Set the path to the data.yaml file
data = '/Users/naveenmalhotra/Downloads/NUMBER_PLATE_DETECTION/DATASET/data.yaml'

# Start training
results = model.train(data=data, epochs=50, imgsz=640)