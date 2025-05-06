import cv2
import torch
import time
import subprocess
import numpy as np
import sys
import os

# Append your AlphaPose repository path (adjust this path as needed)
repo_path = '/srv/pose/bakis/AlphaPose'
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Import the necessary modules from AlphaPose.
# The exact module and class names depend on the repository.
# For this example, assume there's a module "alphapose.models" with a class "PoseEstimationModel".
from alphapose.models import PoseEstimationModel  # adjust as needed

##############################################
# Configuration
##############################################
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 30
RTSP_URL = "rtsp://localhost:8554/live"
MODEL_INPUT_SIZE = (368, 368)  # Example size; adjust as needed

##############################################
# FFmpeg command for RTSP streaming
##############################################
ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{CAM_WIDTH}x{CAM_HEIGHT}",
    "-r", str(TARGET_FPS),
    "-i", "-",  # input from stdin
    "-rtsp_transport", "tcp",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    RTSP_URL
]
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

##############################################
# Load AlphaPose Model and Weights
##############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate your AlphaPose model.
model = PoseEstimationModel()  # adjust parameters as needed
# Load pretrained checkpoint. Replace 'alphapose_checkpoint.pth' with your checkpoint file.
checkpoint_path = 'alphapose_checkpoint.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("AlphaPose weights loaded successfully.")
except Exception as e:
    print("Could not load AlphaPose weights. Running with random weights.", e)
model.to(device)
model.eval()

##############################################
# Initialize the USB Camera
##############################################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the camera!")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

##############################################
# (Optional) Define a function to decode keypoints from model output
# This depends on the AlphaPose model API; assume the model outputs keypoints directly.
def decode_keypoints(model_output):
    # For example, assume model_output is of shape (1, num_keypoints, 3)
    keypoints = model_output.squeeze(0).cpu().numpy()
    return keypoints

##############################################
# Define a COCO skeleton for visualization (17 keypoints)
##############################################
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]
KEYPOINT_COLOR = (0, 255, 0)   # Green
SKELETON_COLOR = (255, 0, 0)   # Blue

##############################################
# Main Loop: Capture, Inference, Draw, and Stream
##############################################
print("Starting AlphaPose pipeline... Press Ctrl+C to exit.")
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocess frame: resize to model input size.
    input_frame = cv2.resize(frame, MODEL_INPUT_SIZE)
    input_frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_array = input_frame_rgb.astype(np.float32) / 255.0
    # Normalize using ImageNet mean and std if required.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_array = (input_array - mean) / std
    input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        model_output = model(input_tensor)
    # Decode keypoints. The function below depends on the model's output format.
    keypoints = decode_keypoints(model_output)  # Expected shape: (num_keypoints, 3)

    # Scale keypoints from MODEL_INPUT_SIZE to original frame size.
    scale_x = CAM_WIDTH / MODEL_INPUT_SIZE[0]
    scale_y = CAM_HEIGHT / MODEL_INPUT_SIZE[1]
    keypoints[:, 0] *= scale_x
    keypoints[:, 1] *= scale_y

    # Draw keypoints.
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.2:
            cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)
    # Draw skeleton lines.
    for (i1, i2) in COCO_SKELETON:
        if keypoints[i1, 2] > 0.2 and keypoints[i2, 2] > 0.2:
            pt1 = (int(keypoints[i1, 0]), int(keypoints[i1, 1]))
            pt2 = (int(keypoints[i2, 0]), int(keypoints[i2, 1]))
            cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)

    # Overlay FPS.
    frame_count += 1
    if frame_count % 10 == 0:
        cur_time = time.time()
        fps = frame_count / (cur_time - prev_time)
        prev_time = cur_time
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    try:
        ffmpeg_process.stdin.write(frame.tobytes())
    except Exception as e:
        print("FFmpeg error:", e)
        break

cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
