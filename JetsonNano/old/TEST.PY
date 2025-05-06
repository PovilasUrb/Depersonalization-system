import cv2
import time
import sys

# --- Configuration ---
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 30

# Set to True to try nvarguscamerasrc, or False to use v4l2src fallback.
USE_NVARGUS = False

gst_cap = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=(int){w}, height=(int){h}, framerate=(fraction){fps}/1, format=(string)YUY2 ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
).format(w=CAM_WIDTH, h=CAM_HEIGHT, fps=TARGET_FPS)

# Output pipeline to stream to mediamtx via UDP.
# This pipeline uses appsrc, converts the frame, encodes with omxh264enc,
# parses and muxes into an MPEG-TS container, and sends via UDP.
# Change 'host' if mediamtx is on a different machine.
gst_out = (
    "appsrc is-live=true ! queue ! videoconvert ! video/x-raw,format=RGBA ! "
    "nvvidconv ! omxh264enc insert-vui=true ! video/x-h264,stream-format=byte-stream ! "
    "h264parse ! mpegtsmux ! udpsink host=127.0.0.1 port=8554"
)

# --- Open the capture pipeline ---
cap = cv2.VideoCapture(gst_cap, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Failed to open camera. If using nvarguscamerasrc, try running as root or check permissions.")
    sys.exit(-1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("Camera opened: {} x {} at {} FPS".format(width, height, fps))

# --- Open the output pipeline ---
writer = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, TARGET_FPS, (CAM_WIDTH, CAM_HEIGHT))
if not writer.isOpened():
    print("Failed to open VideoWriter. Check your GStreamer output pipeline.")
    sys.exit(-2)

print("Starting streaming pipeline. Press Ctrl+C to exit.")
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture error.")
            break

        # (Insert your processing code here, e.g., face detection, pose estimation)
        writer.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print("Streaming at {:.2f} FPS".format(frame_count / elapsed))
except KeyboardInterrupt:
    print("Exiting...")
finally:
    cap.release()
    writer.release()
