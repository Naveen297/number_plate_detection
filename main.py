from ultralytics import YOLO
from PIL import Image
import os

# Load the YOLOv8 model
model = YOLO('/Users/naveenmalhotra/Downloads/NUMBER_PLATE_DETECTION/DATASET/runs/detect/train/weights/best.pt')

# Set the input image folder path
input_folder_path = '/Users/naveenmalhotra/Downloads/NUMBER_PLATE_DETECTION/DATASET/test/images/'

# Set the output image folder path
output_folder_path = '/Users/naveenmalhotra/Downloads/NUMBER_PLATE_DETECTION/DATASET/test/output/'

# Iterate over the files in the input folder
for filename in os.listdir(input_folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Set the input image path
        input_image_path = os.path.join(input_folder_path, filename)

        # Load the input image
        image = Image.open(input_image_path)

        # Run the detection
        results = model(image)

        # Set the output image path
        output_image_path = os.path.join(output_folder_path, filename)

        # Save the resulting image
        results[0].save(output_image_path)
        print(f'Resulting image saved as {output_image_path}')
