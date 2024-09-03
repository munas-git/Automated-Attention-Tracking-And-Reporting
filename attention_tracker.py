# Errors to handle.
# - Empty image payload. (when video is finished).
# - Session stop issue with live stream.
# - Downloading issue (state management).

import os
import cv2
import time
import shutil
import tempfile
import numpy as np
import streamlit as st
from io import BytesIO
import plotly.graph_objs as go
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient

load_dotenv()
RF_API_KEY = os.getenv("RF_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

# Initialize roboflow the inference client.
CLIENT = InferenceHTTPClient(api_url = "https://detect.roboflow.com", api_key = RF_API_KEY)

# Load a font.
try:
    font_fam = "arial.ttf"
    font = ImageFont.truetype(font_fam, 16)
except IOError:
    font = ImageFont.load_default()


distraction_threshold = 3 # Distraction threshold (minimum till recorded as distraction) ... Might adjust "currently distracted logic later to work with this too".
distraction_counter = 0

distraction_start_times = {} # Dictionary for storing each distraction start time.
current_distractions = {} # Dictionary to track current distractions.

# Streamlit app title.
st.title("Real-time Distraction Tracking Dashboard")

# Radio option for choice to upload video or work directly with livestream... (Still need to fix live stream).
upload_option = st.radio("Choose input source:", ("Upload Video", "Use Webcam"))

# Placeholders for the video and distraction counts.
video_placeholder = st.empty()
distraction_count_placeholder = st.empty()

# Columns to place graphs side by side.
col1, col2 = st.columns(2)

with col1:
    st.header("Total Distractions Over Time")
    distraction_graph_placeholder = st.empty()

with col2:
    st.header("Currently Distracted Over Time")
    current_distraction_graph_placeholder = st.empty()

# Lists for storing timeline data for total distractions and currently distracted counts.
distraction_timeline, current_distraction_timeline = [], []

# Initialization of currently distracted count.
currently_distracted_count = 0

# Stop button for manual control in webcam mode
stop_stream_button = st.empty() # Might need to rething current logic.

running = True # Loop control boolean value.

# Check if session state has graph data so that none of them are lost due to streamlit interaction-based refresh.
if "distraction_fig" not in st.session_state:
    st.session_state["distraction_fig"] = None
if "current_distraction_fig" not in st.session_state:
    st.session_state["current_distraction_fig"] = None


def process_frame(frame):
    global distraction_counter
    global distraction_start_times
    global distraction_timeline
    global current_distraction_timeline
    global currently_distracted_count

    # frame >> BGR image >> RGB.
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    temp_file = "temp.jpg"
    image_pil.save(temp_file)

    try:
        result = CLIENT.infer(temp_file, model_id=MODEL_ID)
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return frame

    detected_classes = set()
    currently_distracted_count = 0

    for prediction in result["predictions"]:
        class_name = prediction["class"]
        x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        # Drawing bounding box on frame.
        outline_color = "green" if class_name == "Concentrating" else "red" # Color difference for style and emphasis :)
        draw.rectangle([left, top, right, bottom], outline = outline_color, width = 2)

        # Handling labels; class name, confidence level.... might ingegrate face recognition as well to keep track of who is distracted.
        label = f"{class_name} ({prediction['confidence']:.2f})"
        text_bbox = draw.textbbox((left, top - 20), label, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_bg = [(left, top - text_size[1]), (left + text_size[0], top)]
        draw.rectangle(text_bg, fill = outline_color)
        draw.text((left, top - text_size[1]), label, fill = "white", font = font)

        # Tracking detected classes.
        detected_classes.add(class_name)

        # Updating distraction start times and counter.
        current_time = time.time()
        if class_name not in distraction_start_times:
            distraction_start_times[class_name] = current_time # Adding
        else:
            if distraction_start_times[class_name] is not None:
                if current_time - distraction_start_times[class_name] > distraction_threshold: # Checking if distraction time (duration over frames) is past threshold.
                    if class_name == "Distracted":
                        distraction_counter += 1 # Updating number of distractions only when duration is past threshold.
                        distraction_timeline.append((current_time, distraction_counter))
                        distraction_start_times[class_name] = None  # Reseting after counting to avoid multiple counts of same distraction.

    # Removeing classes that are no longer detected
    detected_classes_list = list(detected_classes)
    for cls in list(distraction_start_times.keys()):
        if cls not in detected_classes_list:
            distraction_start_times.pop(cls, None)

    # Updating currently distracted counts.... Might revisit and integrate this with the whole thresholding logic as well.
    currently_distracted_count = sum(1 for cls in detected_classes if cls == "Distracted")
    current_distraction_timeline.append((time.time(), currently_distracted_count))

    # PIL Image >> OpenCV format.
    annotated_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return annotated_frame

def save_plotly_figure(fig):
    # Save Plotly figure to a BytesIO object... Why? Because it's not necessary to store such temporary data in memory.
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format = "png")
    img_bytes.seek(0)
    return img_bytes

def download_video(video_path, filename):
    video_bytes = open(video_path, "rb").read()
    st.download_button(
        label = "Download Analyzed Video",
        data = video_bytes,
        file_name = filename,
        mime = "video/mp4",
        key = "download_video",
        on_click = st.cache_data.clear
    )


def download_graph(fig, filename):
    img_bytes = save_plotly_figure(fig)
    st.download_button(
        label = f"Download {filename}",
        data = img_bytes,
        file_name = filename,
        mime = "image/png",
        key = f"download_{filename.replace(' ', '_')}",
        on_click = st.cache_data.clear
    )

if upload_option == "Upload Video":
    video_file = st.file_uploader("Upload your video file", type = ["mp4", "avi", "mov"])
    if video_file is not None:
        # Saving uploaded video to temporary file.
        with tempfile.NamedTemporaryFile(delete = False, suffix = '.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name

        cap = cv2.VideoCapture(tmp_file_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Creating directory to store the output video.
        output_dir = tempfile.mkdtemp()
        output_file = os.path.join(output_dir, "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

        # Initialize figures to hold attention metric graphs.
        fig_distraction = go.Figure()
        fig_current_distraction = go.Figure()

        while running:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Failed to grab frame or empty frame received.") # Revisit this later and handle error that occurs when uploauded video file reaches end.
                break

            annotated_frame = process_frame(frame)

            # Writing processed/annotated frame to the output video.
            out.write(annotated_frame)

            # frame: BGR >> RGB.
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Displaying video frame in Streamlit app.
            video_placeholder.image(annotated_frame_rgb, channels = "RGB")

            # Updating distraction count placeholder.... Semi-temporary design; might have to make some UI change to this later.
            distraction_count_placeholder.write(f"Total Distractions: {distraction_counter} | Currently Distracted: {currently_distracted_count}")

            # Updating distraction graph.
            distracted_counts = [x[1] for x in distraction_timeline]
            timestamps = [x[0] for x in distraction_timeline]
            fig_distraction.data = []
            fig_distraction.add_trace(go.Scatter(x = timestamps, y = distracted_counts, mode = "lines+markers"))
            fig_distraction.update_layout(title = "Total Distraction Over Time", xaxis_title = "Time", yaxis_title = "Distraction Count")
            distraction_graph_placeholder.plotly_chart(fig_distraction, use_container_width=True)

            # Update the currently distracted graph
            currently_distracted_counts = [x[1] for x in current_distraction_timeline]
            current_timestamps = [x[0] for x in current_distraction_timeline]
            fig_current_distraction.data = []
            fig_current_distraction.add_trace(go.Scatter(x = current_timestamps, y = currently_distracted_counts, mode = "lines+markers"))
            fig_current_distraction.update_layout(title = "Currently Distracted Over Time", xaxis_title = "Time", yaxis_title = "Current Distraction Count")
            current_distraction_graph_placeholder.plotly_chart(fig_current_distraction, use_container_width = True)

        cap.release()
        out.release()

        # Saving figures to session state for download.... VERY IMPORTANT!!
        st.session_state["distraction_fig"] = fig_distraction
        st.session_state["current_distraction_fig"] = fig_current_distraction

        st.write("Processing complete... Here is the annotated video:")
        download_video(output_file, "annotated_video.mp4")

        # Displaying download buttons only when there actually graphs to download.
        if st.session_state["distraction_fig"]:
            download_graph(st.session_state["distraction_fig"], "total_distractions_graph.png")
        if st.session_state["current_distraction_fig"]:
            download_graph(st.session_state["current_distraction_fig"], "currently_distracted_graph.png")

        # Cleaning up temporary files.
        shutil.rmtree(output_dir)

elif upload_option == "Use Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open webcam.")
        st.stop()

    frame_rate = 20
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    frame_time = 1 / frame_rate

    # Creating directory to store output video.
    output_dir = tempfile.mkdtemp()
    output_file = os.path.join(output_dir, "annotated_webcam.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

    # Initialize figures to hold attention metric graphs.
    fig_distraction = go.Figure()
    fig_current_distraction = go.Figure()

    # Creating stop button.... revisit this later
    stop_stream = st.button("Stop Stream", key = "stop_stream_button")

    while running and not stop_stream:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("Failed to grab frame or empty frame received.")
            continue

        annotated_frame = process_frame(frame)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(annotated_frame_rgb, channels = "RGB")

        out.write(annotated_frame)

        distraction_count_placeholder.write(f"Total Distractions: {distraction_counter} | Currently Distracted: {currently_distracted_count}")

        distracted_counts = [x[1] for x in distraction_timeline]
        timestamps = [x[0] for x in distraction_timeline]
        fig_distraction.data = []
        fig_distraction.add_trace(go.Scatter(x = timestamps, y = distracted_counts, mode = "lines+markers"))
        fig_distraction.update_layout(title = "Total Distraction Over Time", xaxis_title = "Time", yaxis_title = "Distraction Count")
        distraction_graph_placeholder.plotly_chart(fig_distraction, use_container_width = True)


        currently_distracted_counts = [x[1] for x in current_distraction_timeline]
        current_timestamps = [x[0] for x in current_distraction_timeline]
        fig_current_distraction.data = []
        fig_current_distraction.add_trace(go.Scatter(x = current_timestamps, y = currently_distracted_counts, mode = "lines+markers"))
        fig_current_distraction.update_layout(title = "Currently Distracted Over Time", xaxis_title = "Time", yaxis_title = "Current Distraction Count")
        current_distraction_graph_placeholder.plotly_chart(fig_current_distraction, use_container_width = True)

        # Wait for the desired frame time
        time.sleep(frame_time)

    # # Check if the stop button has been pressed
    # stop_stream = st.button("Stop Stream", key = "stop_stream_button")

    st.write("Stream stopped.")
    cap.release()
    out.release()

    st.session_state["distraction_fig"] = fig_distraction
    st.session_state["current_distraction_fig"] = fig_current_distraction

    st.write("Processing complete. Here is the annotated video:")
    download_video(output_file, "annotated_webcam.mp4")

    if st.session_state["distraction_fig"]:
        download_graph(st.session_state["distraction_fig"], "total_distractions_graph.png")
    if st.session_state["current_distraction_fig"]:
        download_graph(st.session_state["current_distraction_fig"], "currently_distracted_graph.png")

    shutil.rmtree(output_dir)