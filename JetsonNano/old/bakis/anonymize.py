import cv2
import numpy as np
import time
import subprocess
import torch

# -----------------------------------------------------------
# Placeholder: Load your Lite-HRNet model.
# You must implement your model definition (e.g., in lite_hrnet.py)
# and adjust the checkpoint loading as needed.
def load_lite_hrnet_model(checkpoint_path):
    # For demonstration, assume you have a model class LiteHRNet in lite_hrnet.py.
    # from lite_hrnet import LiteHRNet
    # model = LiteHRNet()  # Initialize with proper parameters.
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # return model

    # -----
    # For this example, we'll simulate a model.
    # Replace this with your actual model loading code.
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            # Assume output is 17 heatmaps of size 64x48
            self.out_channels = 17
        def forward(self, x):
            batch_size = x.shape[0]
            # Simulate heatmaps: shape [batch, num_keypoints, heatmap_h, heatmap_w]
            return torch.rand(batch_size, self.out_channels, 64, 48)
    model = DummyModel()
    model.eval()
    return model
# -----------------------------------------------------------

# Configuration parameters
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 15

# Expected input resolution for the model (adjust according to your training)
MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 192

# RTSP streaming settings
RTSP_URL = "rtsp://localhost:8554/live"

ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "{}x{}".format(CAM_WIDTH, CAM_HEIGHT),
    "-r", str(TARGET_FPS),
    "-i", "-",  # Input from stdin
    "-rtsp_transport", "tcp",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    RTSP_URL
]

ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# Load the Lite-HRNet model from your checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint_path = "naive_litehrnet_18_coco_256x192.pth"  # Update with your checkpoint file
checkpoint_path = "wider_naive_litehrnet_18_coco_256x192.pth"  # Update with your checkpoint file

model = load_lite_hrnet_model(checkpoint_path)
model.to(device)

# For normalization (using ImageNet stats as an example)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Define a set of skeleton connections (COCO order example for 17 keypoints)
POSE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Initialize the USB camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

prev_time = time.time()
frame_count = 0

print("Starting Lite-HRNet pose estimation pipeline... Press Ctrl+C to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Preprocess the frame:
    # 1. Resize to model input size.
    input_img = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
    # 2. Convert BGR to RGB and normalize.
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_img = (input_img - mean) / std
    # 3. Change to CHW format.
    input_img = np.transpose(input_img, (2, 0, 1))
    # 4. Add batch dimension.
    input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)

    # Run inference with no grad.
    with torch.no_grad():
        output = model(input_tensor)
        # Assume output shape is [1, num_keypoints, out_h, out_w]
    
    # Convert the output to a numpy array.
    heatmaps = output.cpu().numpy()[0]
    num_keypoints, out_h, out_w = heatmaps.shape

    # Decode keypoints: for each keypoint, find the location with maximum activation.
    keypoints = []
    for i in range(num_keypoints):
        heatmap = heatmaps[i]
        _, conf, _, max_loc = cv2.minMaxLoc(heatmap)
        # Scale coordinates from heatmap space to model input space.
        x = max_loc[0] * MODEL_INPUT_WIDTH / float(out_w)
        y = max_loc[1] * MODEL_INPUT_HEIGHT / float(out_h)
        keypoints.append((int(x), int(y)))

    # Scale keypoints back to the original frame size.
    scale_x = float(CAM_WIDTH) / MODEL_INPUT_WIDTH
    scale_y = float(CAM_HEIGHT) / MODEL_INPUT_HEIGHT
    keypoints = [(int(x * scale_x), int(y * scale_y)) for (x, y) in keypoints]

    # Draw keypoints and skeleton on a copy of the original frame.
    output_frame = frame.copy()
    for i, kp in enumerate(keypoints):
        cv2.circle(output_frame, kp, 4, (0, 255, 0), -1)
        cv2.putText(output_frame, str(i), kp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for pair in POSE_PAIRS:
        idx1, idx2 = pair
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            pt1 = keypoints[idx1]
            pt2 = keypoints[idx2]
            cv2.line(output_frame, pt1, pt2, (255, 0, 0), 2)

    # Optional: Overlay FPS.
    frame_count += 1
    if frame_count % 10 == 0:
        current_time = time.time()
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0
        cv2.putText(output_frame, "FPS: {:.2f}".format(fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Send the processed frame to FFmpeg for RTSP streaming.
    try:
        ffmpeg_process.stdin.write(output_frame.tobytes())
    except Exception as e:
        print("FFmpeg error:", e)
        break

cap.release()
if ffmpeg_process.stdin:
    ffmpeg_process.stdin.close()
ffmpeg_process.wait()
