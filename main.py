import os
from dotenv import load_dotenv
import cv2  # OpenCV for video handling and drawing
import datetime # For formatting timestamp
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging

# Suppress most of Ultralytics' default logging
LOGGER.setLevel(logging.WARNING) # Or logging.ERROR

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH environment variable is not set.")

VIDEO_PATH = os.getenv("VIDEO_PATH")
if not VIDEO_PATH:
    raise ValueError("VIDEO_PATH environment variable is not set.")

# Load the YOLO model
model = YOLO(MODEL_PATH)

# --- Determine the class ID for 'person' or 'human' ---
person_class_id = -1
human_class_names_to_check = ['person', 'human'] # Common names
for class_id, name in model.names.items():
    if name.lower() in human_class_names_to_check:
        person_class_id = class_id
        print(f"Found human-like class: '{name}' with ID: {person_class_id}")
        break

if person_class_id == -1:
    print("Warning: Could not find a 'person' or 'human' class in model.names. Human count will be 0.")
    # Decide how to handle this: raise error, or proceed with 0 counts.
    # For this example, we'll proceed, but human_count will always be 0.

# Open the video file with OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Warning: Could not determine FPS from video. Assuming 30 FPS for skipping logic.")
    fps = 30 # Fallback FPS

print(f"Video FPS: {fps:.2f}")
print(f"Processing approximately one frame per second...")

# --- Frame skipping logic ---
# We want to process roughly one frame per second.
# So, we'll skip `fps - 1` frames after processing one.
# More accurately, we'll process the first frame of each new second.

frame_count = 0
last_processed_second_marker = -1 # To track which second we last processed a frame for

print("\n--- Detection Results (processing one frame per second) ---")

while cap.isOpened():
    ret, frame = cap.read() # Read a frame from the video
    if not ret:
        break # End of video or error

    current_second_marker = int(frame_count / fps)

    # Only process if this frame belongs to a new second
    if current_second_marker > last_processed_second_marker:
        # This is the frame we'll process for this second
        last_processed_second_marker = current_second_marker

        # Perform inference on the current frame
        # Note: model(frame) returns a list of Results objects (usually one for a single image)
        results_list = model(frame, verbose=False) # verbose=False for model's own logging
        
        human_count = 0
        
        # The result for the single frame is the first element in the list
        if results_list:
            result = results_list[0] 
            
            # Ensure result.boxes is not None before iterating
            if result.boxes:
                for box in result.boxes:
                    # box.cls is a tensor, get the first class ID as int
                    class_id = int(box.cls[0])
                    
                    if class_id == person_class_id:
                        human_count += 1
                        # Get bounding box coordinates
                        xyxy = box.xyxy[0].cpu().numpy() # [xmin, ymin, xmax, ymax]
                        # Draw rectangle
                        cv2.rectangle(frame, 
                                      (int(xyxy[0]), int(xyxy[1])), 
                                      (int(xyxy[2]), int(xyxy[3])), 
                                      (0, 255, 0), 2) # Green box for humans
                        # Add label
                        cv2.putText(frame, model.names[class_id], 
                                    (int(xyxy[0]), int(xyxy[1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Format timestamp
        td = datetime.timedelta(seconds=current_second_marker)
        formatted_timestamp = str(td)

        print(f"Timestamp: {formatted_timestamp} (Source Frame: {frame_count+1}), Humans Detected: {human_count}")
        
        # Display the frame (with detections from the processed frame)
        cv2.imshow('Processed Frame (1 FPS)', frame)

    else:
        # If not processing this frame, you could still display it without detections
        # or do nothing to speed up even more by not rendering skipped frames.
        # For this example, we only show the processed frame.
        pass


    frame_count += 1

    # Allow breaking the loop with 'q'
    # cv2.waitKey(1) is non-blocking if a frame is displayed.
    # If no frame is displayed in the loop (e.g., if you comment out cv2.imshow),
    # waitKey might block indefinitely or behave differently.
    # A small delay helps keep the window responsive.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("\nProcessing complete.")