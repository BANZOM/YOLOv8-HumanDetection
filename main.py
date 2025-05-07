import os
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH environment variable is not set.")

from ultralytics import YOLO
model = YOLO(MODEL_PATH)

VIDEO_PATH = os.getenv("VIDEO_PATH")
if not VIDEO_PATH:
    raise ValueError("VIDEO_PATH environment variable is not set.")

results = model(VIDEO_PATH, stream=True)

for result in results:
    # Print results
    print(result)
    # Print results with labels
    print(result.names)
    # Print results with bounding boxes
    print(result.boxes)
    # Print results with masks
    print(result.masks)
    # Print results with keypoints
    print(result.keypoints)
    # Print results with segmentation
    print(result.segmentation)

    