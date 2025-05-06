#!/usr/bin/env python3
import cv2
import numpy as np
import subprocess
import time
from tflite_runtime.interpreter import Interpreter

# === Configuration ===
MODEL_PATH    = "/opt/yolo11n-seg_float16.tflite"  # Your YOLOv11-seg TFLite model
# We will use the model's default input shape (should be [1, 640, 640, 3])
# (Do not force a resize if the model is fixed)

# Streaming output resolution – we choose 640x640 for consistency.
OUTPUT_WIDTH  = 640
OUTPUT_HEIGHT = 640
TARGET_FPS    = 15          # Target frame rate for processing
SEG_THRESHOLD = 0.5         # Threshold for segmentation mask

# RTSP output endpoint – ensure your RTSP server (e.g. Mediamtx) is running and accepting a publisher
RTSP_OUTPUT   = "rtsp://localhost:8554/live"

# === Initialize TFLite Interpreter ===
print("Initializing segmentation model from:", MODEL_PATH)
interpreter = Interpreter(model_path=MODEL_PATH)
default_shape = interpreter.get_input_details()[0]['shape']
print("Using model's default input shape:", default_shape)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded.")
print("Input details:", input_details)
print("Output details:", output_details)

# Get the model input size from the default shape
model_input_height = input_details[0]['shape'][1]
model_input_width  = input_details[0]['shape'][2]
print("Model expects input of size {}x{}".format(model_input_width, model_input_height))

# === Preprocessing Function ===
def preprocess(frame):
    """
    Resize a copy of the frame to the model's input size using INTER_AREA,
    convert from BGR to RGB, cast to float32 and normalize to [0,1],
    then add the batch dimension.
    """
    resized = cv2.resize(frame, (model_input_width, model_input_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0
    return input_data

# === Postprocessing Function ===
def postprocess(output_data, frame):
    """
    Process the model output to extract a segmentation mask,
    resize the mask to the output frame size, and overlay a translucent green mask
    on regions where the segmentation probability exceeds SEG_THRESHOLD.
    
    We assume that the model's second output tensor (index 1) has shape
    [1, 160, 160, 32] and that channel 0 of this tensor contains the segmentation mask.
    """
    try:
        # Extract the second output tensor
        seg_output = output_data[1]  # shape: [1, 160, 160, 32]
        # Take the first channel as the segmentation probability map
        seg_map = seg_output[0, :, :, 0]  # shape: [160, 160]
    except Exception as e:
        print("Error extracting segmentation mask:", e)
        return frame

    # Create a binary mask by thresholding
    binary_mask = (seg_map >= SEG_THRESHOLD).astype(np.uint8) * 255
    # Resize the binary mask to match the output resolution
    mask_resized = cv2.resize(binary_mask, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # Create a green color mask
    color_mask = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    color_mask[:] = (0, 255, 0)
    
    # Overlay the color mask on the original frame where the binary mask is set,
    # using an alpha transparency factor.
    alpha = 0.5
    overlay = frame.copy()
    overlay[mask_resized > 0] = cv2.addWeighted(frame[mask_resized > 0],
                                                 1 - alpha,
                                                 color_mask[mask_resized > 0],
                                                 alpha,
                                                 0)
    return overlay

# === Capture Video from USB Camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open USB camera!")
    exit(1)

# Set camera properties (if supported by your camera)
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
    '-i', '-',  # Read raw video from stdin
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-b:v', '1M',   # Bitrate (adjust as needed)
    '-g', '50',     # Keyframe interval
    '-f', 'rtsp',
    RTSP_OUTPUT
]
ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
print("Streaming processed video to RTSP endpoint:", RTSP_OUTPUT)

# === Main Processing Loop ===
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

        # For inference, resize a copy of the captured frame to the model's input size
        # and keep the original for overlay.
        inference_frame = cv2.resize(frame, (model_input_width, model_input_height), interpolation=cv2.INTER_AREA)
        input_data = preprocess(frame)  # Alternatively, preprocess(inference_frame)
        
        # Run segmentation inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']), interpreter.get_tensor(output_details[1]['index'])
        
        # Use the original captured frame (resized to output resolution) for overlay
        output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        annotated_frame = postprocess(output_data, output_frame.copy())

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

        # Enforce roughly the target frame rate
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
