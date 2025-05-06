import cv2
import time
import subprocess
import numpy as np
from openpifpaf import Predictor

# ---------------------------
# Configuration
# ---------------------------
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 30
RTSP_URL = "rtsp://localhost:8554/live"

# ---------------------------
# FFmpeg command for RTSP streaming
# ---------------------------
ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{CAM_WIDTH}x{CAM_HEIGHT}",
    "-r", str(TARGET_FPS),
    "-i", "-",  # read input from stdin
    "-rtsp_transport", "tcp",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    RTSP_URL
]
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# ---------------------------
# Initialize the USB Camera
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the camera!")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# ---------------------------
# Initialize OpenPifPaf Predictor
# ---------------------------
# This will download the pretrained 'resnet50' weights on first run.
predictor = Predictor(checkpoint='resnet50')

# ---------------------------
# Define COCO skeleton connections (indices for 17 keypoints)
# Standard COCO keypoints:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Colors for drawing (BGR)
KEYPOINT_COLOR = (0, 255, 0)   # Green for keypoints
SKELETON_COLOR = (255, 0, 0)   # Blue for skeleton lines

print("Starting OpenPifPaf pose estimation pipeline... Press Ctrl+C to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame from BGR to RGB for OpenPifPaf processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose estimation; predictor.numpy() returns a tuple: (predictions, gt_annotations, meta)
    predictions, _, _ = predictor.numpy(rgb_frame)
    
    # Draw predictions on the original frame (still in BGR)
    for pred in predictions:
        # pred.data is an array of shape (num_keypoints, 3) where each row is [x, y, confidence]
        keypoints = pred.data
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.2:
                cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)
        # Draw skeleton lines between keypoints if both have sufficient confidence.
        for (i1, i2) in COCO_SKELETON:
            if keypoints[i1, 2] > 0.2 and keypoints[i2, 2] > 0.2:
                pt1 = (int(keypoints[i1, 0]), int(keypoints[i1, 1]))
                pt2 = (int(keypoints[i2, 0]), int(keypoints[i2, 1]))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)

    # Optionally, overlay FPS text.
    cv2.putText(frame, f"FPS: {TARGET_FPS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the annotated frame to FFmpeg for RTSP streaming.
    try:
        ffmpeg_process.stdin.write(frame.tobytes())
    except Exception as e:
        print("FFmpeg error:", e)
        break

cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
