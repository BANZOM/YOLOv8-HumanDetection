import datetime
import os
import cv2
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Load the model path from the environment variable
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH environment variable is not set.")

# Import YOLO model from the ultralytics library
from ultralytics import YOLO
model = YOLO(MODEL_PATH)  # Load the YOLO model using the specified path

# Load the video path from the environment variable
VIDEO_PATH = os.getenv("VIDEO_PATH")
if not VIDEO_PATH:
    raise ValueError("VIDEO_PATH environment variable is not set.")

# Open the video file using OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

# Get the frames per second (FPS) and total frame count of the video
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()  # Release the video capture object

# Handle cases where FPS could not be determined
if fps == 0:
    print("Warning: Could not determine FPS, timestamp calculation might be inaccurate. Assuming 30 FPS.")
    fps = 30  # Fallback FPS value

# Print video metadata
print(f"Video FPS: {fps}, Total Frames: {total_frames}")
print(f"Model class names: {model.names}")

# Define human-related class names to detect
human_class_names = ['person', 'human']
human_class_ids = None

# Identify the class IDs corresponding to human-related classes
for class_id, class_name in model.names.items():
    if class_name.lower() in human_class_names:
        if human_class_ids is None:
            human_class_ids = []
        human_class_ids.append(class_id)

# Raise an error if no human-related class IDs are found
if not human_class_ids:
    raise ValueError("No human class IDs found in the model.")

# Print the detected human class IDs
print(f"Human class IDs: {human_class_ids}")

# Perform object detection on the video in streaming mode
results = model(VIDEO_PATH, stream=True)

# Iterate through the detection results frame by frame
for i, result in enumerate(results):
    current_frame_number = i + 1  # Frame number (1-indexed)
    timestamp_seconds = current_frame_number / fps  # Calculate timestamp in seconds
    
    # Format the timestamp as H:M:S.ms
    td = datetime.timedelta(seconds=timestamp_seconds)
    formatted_timestamp = str(td)

    human_count = 0  # Initialize human count for the current frame
    frame = result.orig_img  # Extract the original frame from the result

    # Check if there are any detections in the current frame
    if result.boxes:
        for box in result.boxes:
            # Extract the class IDs from the detection box
            class_ids = box.cls.int().tolist()
            # Check if any detected class ID matches the human class IDs
            if any(class_id in human_class_ids for class_id in class_ids):
                human_count += 1  # Increment human count
                # Get bounding box coordinates as a numpy array
                xyxy = box.xyxy[0].cpu().numpy()
                # Draw a bounding box around the detected human
                cv2.rectangle(frame, 
                              (int(xyxy[0]), int(xyxy[1])), 
                              (int(xyxy[2]), int(xyxy[3])), 
                              (255, 0, 0), 2)
                # Add a label "Human" above the bounding box
                cv2.putText(frame, "Human", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with detections (press 'q' to exit)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Print the timestamp, frame number, and human count for the current frame
    print(f"Timestamp: {formatted_timestamp} (Frame: {current_frame_number}), Humans Detected: {human_count}")
