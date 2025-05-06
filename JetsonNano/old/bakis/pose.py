import cv2
import torch
import torch.nn as nn  # Make sure to import torch.nn as nn
import torchvision
import time
import subprocess
import numpy as np

##############################################
# Minimal Pose Estimation Model Using MobileNetV2 Backbone
##############################################
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class LiteHRNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(LiteHRNet, self).__init__()
        # Initial convolution: reduce resolution by half.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # Basic layers
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        # Deconvolution (upsampling) layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        # Final convolution to output heatmaps for each keypoint.
        self.out_conv = nn.Conv2d(32, num_keypoints, kernel_size=1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (B,32,H/2,W/2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn2(self.deconv1(x)))  # Upsample to ~H/4
        x = self.relu(self.bn3(self.deconv2(x)))  # Upsample to ~H/2
        x = self.relu(self.bn4(self.deconv3(x)))  # Upsample to ~original resolution
        x = self.out_conv(x)  # (B, num_keypoints, H_out, W_out)
        return x

##############################################
# Helper Function to Decode Keypoints
##############################################
def get_max_preds(batch_heatmaps):
    # Expects shape: (B, K, H, W)
    B, K, H, W = batch_heatmaps.shape
    preds = np.zeros((B, K, 2))
    maxvals = np.zeros((B, K, 1))
    for b in range(B):
        for k in range(K):
            hm = batch_heatmaps[b, k, :, :]
            flat_index = np.argmax(hm)
            maxval = np.max(hm)
            maxvals[b, k, 0] = maxval
            preds[b, k, 0] = flat_index % W
            preds[b, k, 1] = flat_index // W
    return preds, maxvals

##############################################
# Configuration Parameters
##############################################
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 30
# Model input size expected by the model (width, height)
MODEL_INPUT_SIZE = (368, 368)
# RTSP stream URL
RTSP_URL = "rtsp://localhost:8554/live"

##############################################
# FFmpeg Setup for RTSP Streaming
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
# Model Loading
##############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiteHRNet(num_keypoints=17)
checkpoint_path = 'naive_litehrnet_18_coco_256x192.pth'  # Replace with your checkpoint file

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # If checkpoint is a dict with a state_dict key, extract it.
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")
except Exception as e:
    print("Could not load weights. Running with random weights.", e)
model.to(device)
model.eval()

##############################################
# Camera Initialization
##############################################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the camera!")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

##############################################
# Define COCO Skeleton for 17 Keypoints
##############################################
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]
KEYPOINT_COLOR = (0, 255, 0)   # Green for keypoints
SKELETON_COLOR = (255, 0, 0)   # Blue for skeleton lines

##############################################
# Main Loop: Capture, Inference, Draw, and Stream
##############################################
print("Starting pose estimation pipeline... Press Ctrl+C to exit.")
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocess frame: resize to MODEL_INPUT_SIZE.
    input_frame = cv2.resize(frame, MODEL_INPUT_SIZE)
    # Convert BGR to RGB.
    input_frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    # Normalize pixel values.
    input_array = input_frame_rgb.astype(np.float32) / 255.0
    # Normalize using ImageNet mean and std.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_array = (input_array - mean) / std
    # Convert to tensor: shape (1, 3, H, W).
    input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)  # Output shape: (1, 17, H, W)
    heatmaps = output.cpu().numpy()[0]  # Now heatmaps shape: (17, H, W)
    for i in range(heatmaps.shape[0]):
        print(f"Channel {i} max confidence: {np.max(heatmaps[i])}")

    # If heatmaps is 3D, add a batch dimension.
    if heatmaps.ndim == 3:
        heatmaps = np.expand_dims(heatmaps, 0)

    preds, maxvals = get_max_preds(heatmaps)  # preds: (1, 17, 2), maxvals: (1, 17, 1)
    keypoints = preds[0]  # (17, 2)
    confidences = maxvals[0, :, 0]  # (17,)
    
    # Scale keypoints from heatmap resolution to MODEL_INPUT_SIZE.
    # We assume the output resolution is MODEL_INPUT_SIZE/4.
    heatmap_size = (MODEL_INPUT_SIZE[0] // 4, MODEL_INPUT_SIZE[1] // 4)
    scale_factor_x = MODEL_INPUT_SIZE[0] / heatmap_size[0]
    scale_factor_y = MODEL_INPUT_SIZE[1] / heatmap_size[1]
    keypoints[:, 0] *= scale_factor_x
    keypoints[:, 1] *= scale_factor_y

    # Scale keypoints from MODEL_INPUT_SIZE to original frame size.
    scale_x = CAM_WIDTH / MODEL_INPUT_SIZE[0]
    scale_y = CAM_HEIGHT / MODEL_INPUT_SIZE[1]
    keypoints[:, 0] *= scale_x
    keypoints[:, 1] *= scale_y

    # Draw keypoints.
    for i, (x, y) in enumerate(keypoints):
        if confidences[i] > 0.2:
            cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)
    # Draw skeleton lines.
    for (i1, i2) in COCO_SKELETON:
        if confidences[i1] > 0.2 and confidences[i2] > 0.2:
            pt1 = (int(keypoints[i1, 0]), int(keypoints[i1, 1]))
            pt2 = (int(keypoints[i2, 0]), int(keypoints[i2, 1]))
            cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)

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
