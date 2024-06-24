from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('/Users/naveenmalhotra/Downloads/NUMBER_PLATE_DETECTION/DATASET/runs/detect/train/weights/best.pt')

# Set the video file path
video_path = '/Users/naveenmalhotra/Downloads/NUMBER_PLATE_DETECTION/DATASET/sample.mp4'

# Create a video capture object
cap = cv2.VideoCapture(video_path)

# Get the original video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the desired width and height
new_width = 800
new_height = 540

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (new_width, new_height))

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the desired width and height
    frame = cv2.resize(frame, (new_width, new_height))

    # Run the detection on the frame
    results = model(frame)

    # Draw the bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow('License Plate Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()