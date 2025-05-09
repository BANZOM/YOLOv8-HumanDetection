import streamlit as st
import os
import cv2
import datetime
import tempfile
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables (primarily for MODEL_PATH)
load_dotenv()
MODEL_PATH_ENV = os.getenv("MODEL_PATH")

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLOv8 Human Detection",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading (Cached) ---
@st.cache_resource # Cache the model loading for efficiency
def load_yolo_model(model_path):
    if not model_path:
        st.error("MODEL_PATH environment variable is not set. Please set it in your .env file.")
        return None
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_yolo_model(MODEL_PATH_ENV)

# --- Helper function to find person class ID ---
@st.cache_data # Can cache this if model.names doesn't change
def get_person_class_id(_model):
    if _model is None:
        return -1
    person_class_id = -1
    human_class_names_to_check = ['person', 'human']
    for class_id, name in _model.names.items():
        if name.lower() in human_class_names_to_check:
            person_class_id = class_id
            break
    if person_class_id == -1:
        st.warning("Could not find a 'person' or 'human' class in the loaded model's class names.")
    return person_class_id

if model:
    PERSON_CLASS_ID = get_person_class_id(model)
else:
    PERSON_CLASS_ID = -1


# --- Video Processing Function ---
def process_video_streamlit(video_path, yolo_model, person_cls_id, progress_bar_placeholder, video_placeholder, stats_placeholder):
    """
    Processes the video, yielding results for Streamlit to display.
    """
    if not yolo_model or person_cls_id == -1:
        st.error("Model not loaded or person class ID not found. Cannot process video.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.warning("Could not determine video FPS. Assuming 30 FPS.")
        fps = 30 # Fallback
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_seconds = total_frames / fps if fps > 0 else 0

    frame_count = 0
    last_processed_second_marker = -1
    human_counts_log = [] # To store (timestamp_seconds, count)

    st.markdown("---") # Visual separator

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_second_marker = int(frame_count / fps) # Integer second

        # Process one frame per unique second
        if current_second_marker > last_processed_second_marker:
            last_processed_second_marker = current_second_marker
            
            # Perform inference
            results_list = yolo_model(frame, verbose=False, device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu') # Use GPU if available
            
            processed_frame_for_display = frame.copy() # Work on a copy for drawing
            human_count_this_second = 0

            if results_list and results_list[0].boxes:
                for box in results_list[0].boxes:
                    class_id = int(box.cls[0])
                    if class_id == person_cls_id:
                        human_count_this_second += 1
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        # Draw bounding box
                        cv2.rectangle(processed_frame_for_display,
                                      (int(xyxy[0]), int(xyxy[1])),
                                      (int(xyxy[2]), int(xyxy[3])),
                                      (34, 139, 34), 2) # Forest green
                        
                        # Add label
                        label = f"{yolo_model.names[class_id]} ({box.conf[0]:.2f})"
                        cv2.putText(processed_frame_for_display, label,
                                    (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA) # White text
                        cv2.putText(processed_frame_for_display, label,
                                    (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34,139,34), 1, cv2.LINE_AA) # Green outline

            # Update Streamlit elements
            td = datetime.timedelta(seconds=current_second_marker)
            formatted_timestamp = str(td)
            
            # Convert frame to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(processed_frame_for_display, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, caption=f"Processed Frame from ~{formatted_timestamp}")
            
            stats_placeholder.markdown(f"**Timestamp:** `{formatted_timestamp}` (Frame: {frame_count+1})<br>"
                                       f"**Humans Detected in this frame:** `{human_count_this_second}`",
                                       unsafe_allow_html=True)
            human_counts_log.append({"Time (s)": current_second_marker, "Humans": human_count_this_second, "Timestamp": formatted_timestamp})

            # Update progress bar
            if total_duration_seconds > 0:
                progress_bar_placeholder.progress(min(1.0, current_second_marker / total_duration_seconds))
            else:
                progress_bar_placeholder.progress(0) # Cannot determine progress

        frame_count += 1
        
        # Allow early exit if the Streamlit app is re-run or tab is closed (helps manage resources)
        # This is a bit of a hack, Streamlit doesn't have a direct "stop" button for backend loops easily
        if cv2.waitKey(1) & 0xFF == ord('q'): # Not really effective in Streamlit, but good practice
            break


    cap.release()
    if total_duration_seconds > 0:
        progress_bar_placeholder.progress(1.0) # Mark as complete
    return human_counts_log

# --- Main UI ---
st.title("🚶 YOLOv8 Real-time Human Detection")
st.markdown("""
Upload a video file, and this app will process it (approximately one frame per second of video time) 
to detect and count humans using a YOLOv8 model.
""")

if not model:
    st.warning("YOLO Model could not be loaded. Please check `MODEL_PATH` in your `.env` file and ensure the model file exists.")
    st.stop()
if PERSON_CLASS_ID == -1 and model:
    st.warning(f"The loaded model does not seem to have a 'person' or 'human' class. Detected classes: `{model.names}`. Human counting will not work correctly.")


# Sidebar for Upload and Controls
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    
    # Placeholder for processing rate if you want to make it configurable later
    # processing_rate_fps = st.slider("Processing Rate (video FPS to analyze)", 1, int(model_fps_cap if model_fps_cap else 30), 1)

    start_processing_button = st.button("Start Processing", type="primary", disabled=(uploaded_file is None))

if start_processing_button and uploaded_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name
    
    st.info(f"Processing video: `{uploaded_file.name}`. This might take a moment...")

    # Placeholders for dynamic content
    progress_bar_placeholder = st.empty()
    progress_bar_placeholder.progress(0) # Initial progress

    col1, col2 = st.columns([2,1]) # Video on left, stats on right
    with col1:
        video_placeholder = st.empty()
        video_placeholder.info("Video processing will appear here...")
    with col2:
        stats_placeholder = st.empty()
        stats_placeholder.info("Detection stats will appear here...")

    try:
        detection_log = process_video_streamlit(
            temp_video_path, 
            model, 
            PERSON_CLASS_ID,
            progress_bar_placeholder,
            video_placeholder,
            stats_placeholder
        )
        st.success("Video processing complete!")
        
        if detection_log:
            st.subheader("📊 Detection Summary")
            
            # Create a DataFrame for easier analysis and charting
            import pandas as pd
            df_log = pd.DataFrame(detection_log)
            
            # Display metrics
            max_humans = df_log["Humans"].max() if not df_log.empty else 0
            avg_humans = df_log["Humans"].mean() if not df_log.empty else 0
            
            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric("Max Humans Detected (in any processed second)", int(max_humans))
            col_metric2.metric("Avg Humans Detected (per processed second)", f"{avg_humans:.2f}")

            st.line_chart(df_log.rename(columns={'Humans': 'Human Count'}).set_index('Timestamp')['Human Count'])
            
            with st.expander("View Raw Detection Log"):
                st.dataframe(df_log)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        progress_bar_placeholder.empty() # Clear progress bar after completion or error

elif uploaded_file is None and 'start_processing_button_clicked_once' not in st.session_state :
    st.info("Upload a video file and click 'Start Processing' to begin.")

# This helps to keep the "Start Processing" button state correctly
if start_processing_button:
    st.session_state['start_processing_button_clicked_once'] = True
