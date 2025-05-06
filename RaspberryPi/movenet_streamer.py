#!/usr/bin/env python3
import cv2
import numpy as np
import subprocess
import time
from tflite_runtime.interpreter import Interpreter

# === Configuration ===
MODEL_PATH    = "/opt/movenet_lightning.tflite"  # Path to your MoveNet multipose model
INPUT_HEIGHT  = 192    # Desired model input height
INPUT_WIDTH   = 256    # Desired model input width

# RTSP output endpoint (Mediamtx should be running and configured to accept this)
RTSP_OUTPUT   = "rtsp://localhost:8554/live"

# Output video dimensions (used for both display and streaming)
OUTPUT_WIDTH  = 640
OUTPUT_HEIGHT = 480

TARGET_FPS    = 10     # Target processing frame rate
CONF_THRESHOLD = 0.3   # Confidence threshold for drawing keypoints

# Define COCO skeleton connections (indices per COCO keypoint order)
skeleton = [
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    (0, 5),   # nose -> left_shoulder
    (0, 6),   # nose -> right_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    (5, 6),   # left_shoulder <-> right_shoulder
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12), # left_hip <-> right_hip
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16)  # right_knee -> right_ankle
]

# === Initialize TFLite Interpreter ===
print("Initializing model from:", MODEL_PATH)
interpreter = Interpreter(model_path=MODEL_PATH)
new_shape = [1, INPUT_HEIGHT, INPUT_WIDTH, 3]
print("Resizing input tensor to:", new_shape)
interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], new_shape)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded.")
print("Input details:", input_details)
print("Output details:", output_details)

# === Preprocessing Function ===
def preprocess(frame):
    # Resize frame to the model's expected input dimensions using INTER_AREA
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    # Convert from BGR (OpenCV default) to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Expand dims to form a batch of 1, cast to uint8 (assumes quantized model)
    input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
    return input_data

# === Keypoint Extraction Function ===
def extract_best_detection(output_data):
    """
    Given the model output of shape (1, 6, 56), select the detection with the highest overall score.
    Assumes:
      - The first 51 values in each detection correspond to keypoints (17 keypoints * 3 values)
      - The overall detection score is at index 55.
    Returns a (17, 3) numpy array.
    """
    detections = output_data[0]  # shape: (6, 56)
    overall_scores = detections[:, 55]
    best_idx = int(np.argmax(overall_scores))
    best_detection = detections[best_idx]  # shape: (56,)
    keypoints = best_detection[:51].reshape(17, 3)
    return keypoints

# === Postprocessing Function ===
def postprocess(output_data, frame):
    """
    Extracts keypoints from the model output, draws keypoints and connects them with lines
    according to the skeleton. Keypoints with confidence >= CONF_THRESHOLD are drawn.
    The normalized coordinates are scaled to OUTPUT_WIDTH and OUTPUT_HEIGHT.
    """
    try:
        keypoints = extract_best_detection(output_data)
    except Exception as e:
        print("Error extracting keypoints:", e)
        return frame

    points = []
    for kp in keypoints:
        try:
            y, x, score = kp  # Expected order: (y, x, score)
        except Exception as e:
            continue
        px = int(x * OUTPUT_WIDTH)
        py = int(y * OUTPUT_HEIGHT)
        points.append((px, py, score))
        if score >= CONF_THRESHOLD:
            cv2.circle(frame, (px, py), 4, (0, 255, 0), thickness=-1)

    # Connect keypoints using the COCO skeleton
    for (i, j) in skeleton:
        if i < len(points) and j < len(points):
            (xi, yi, si) = points[i]
            (xj, yj, sj) = points[j]
            if si >= CONF_THRESHOLD and sj >= CONF_THRESHOLD:
                cv2.line(frame, (xi, yi), (xj, yj), (255, 0, 0), thickness=2)
    return frame

# === Main Loop Using USB Camera as Input ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open USB camera!")
    exit(1)

# Optionally set camera resolution and FPS (or resize later)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# === Set Up FFmpeg Process for RTSP Streaming ===
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}',
    '-r', str(TARGET_FPS),
    '-i', '-',  # Read from stdin
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-b:v', '1M',   # Bitrate (adjust as needed)
    '-g', '50',     # Keyframe interval
    '-f', 'rtsp',
    'rtsp://localhost:8554/live'
]

ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
print("Streaming processed video to RTSP endpoint rtsp://localhost:8554/live ...")

frame_interval = 1.0 / TARGET_FPS
frame_count = 0
last_time = time.time()

try:
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from USB camera.")
            break

        # Ensure frame is the correct resolution
        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        # Preprocess the frame for inference
        input_data = preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Postprocess: draw keypoints and connect them with lines
        annotated_frame = postprocess(output_data, frame.copy())

        # Write the annotated frame to FFmpeg's stdin
        try:
            ffmpeg.stdin.write(annotated_frame.tobytes())
            ffmpeg.stdin.flush()
        except BrokenPipeError:
            print("FFmpeg pipe broken")
            break

        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps_current = frame_count / (current_time - last_time)
            print(f"Current FPS: {fps_current:.2f}")
            frame_count = 0
            last_time = current_time

        # Enforce approximately the target frame rate
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
