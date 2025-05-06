import sys
import os
import cv2
import time
import numpy as np
import torch

################################################################################
# GLOBAL VARIABLES: TWEAKS, TOGGLES, & CONFIG
################################################################################

SHOW_FPS = True

###################### FACE DETECTION SETTINGS ######################
FACE_DETECTOR_ENABLED = True
FACE_MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to .caffemodel
FACE_CONFIG_FILE = "deploy.prototxt"                          # Path to .prototxt
FACE_CONFIDENCE_THRESHOLD = 0.5

KALMAN_ENABLED = True
IOU_THRESHOLD = 0.3
MAX_MISSED_FRAMES = 5
RED_BOX_SCALE = 1.4

###################### POSE ESTIMATION SETTINGS ######################
POSE_ESTIMATOR_ENABLED = True
REPO_PATH = '/srv/bakis/lightweight-human-pose-estimation.pytorch'
POSE_CHECKPOINT_PATH = "checkpoint_iter_370000.pth"
POSE_HEIGHT_SIZE = 160
POSE_UPSAMPLE_RATIO = 1
POSE_FP16 = False
POSE_CONFIDENCE_THRESHOLD = 0.6
TRACK_POSE = True
SMOOTH_POSE = True
POSE_FACE_ANONIMYZE = True

###################### VIDEO STREAM SETTINGS ######################
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 15

# GStreamer capture pipeline for a USB camera (v4l2src).
# Adjust device=/dev/video0 if needed, or width/height/fps if your camera differs.
gst_in = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=(int){w}, height=(int){h}, framerate=(fraction){fps}/1, format=(string)YUY2 ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
).format(w=CAM_WIDTH, h=CAM_HEIGHT, fps=TARGET_FPS)

# GStreamer output pipeline to send processed frames via UDP in MPEG-TS.
# omxh264enc is Jetson Nano's hardware encoder; adjust 'host' for your mediamtx instance.
gst_out = (
    "appsrc is-live=true ! queue ! videoconvert ! video/x-raw,format=RGBA ! "
    "nvvidconv ! omxh264enc insert-vui=true ! video/x-h264,stream-format=byte-stream ! "
    "h264parse ! mpegtsmux ! "
    "udpsink host=127.0.0.1 port=8554"
)

################################################################################
# FACE DETECTION & TRACKING HELPERS
################################################################################

def iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes: (startX, startY, endX, endY)."""
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    denom = boxAArea + boxBArea - interArea
    if denom == 0:
        return 0
    return interArea / float(denom)

class FaceTracker:
    def __init__(self, bbox, tracker_id):
        self.id = tracker_id
        self.last_bbox = bbox  # (startX, startY, endX, endY)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.time_since_update = 0
        if KALMAN_ENABLED:
            self.kalman = cv2.KalmanFilter(4, 2)
            self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)
            self.kalman.transitionMatrix  = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], np.float32)
            self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
            init_state = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kalman.statePre = init_state.copy()
            self.kalman.statePost = init_state.copy()
            self.predicted_center = (cx, cy)
        else:
            self.predicted_center = (cx, cy)

    def advance(self):
        """Advance the tracker by one frame."""
        self.time_since_update += 1
        if KALMAN_ENABLED:
            pred = self.kalman.predict()
            self.predicted_center = (pred[0, 0], pred[1, 0])

    def update(self, bbox):
        """Update tracker with new detection."""
        self.last_bbox = bbox
        self.time_since_update = 0
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        if KALMAN_ENABLED:
            meas = np.array([[np.float32(cx)], [np.float32(cy)]])
            self.kalman.correct(meas)
            self.predicted_center = (cx, cy)
        else:
            self.predicted_center = (cx, cy)

    def mark_missed(self):
        """Mark the tracker as having missed this frame."""
        self.time_since_update += 1

    def get_bbox(self):
        """Compute bounding box from the predicted center."""
        cx, cy = self.predicted_center
        startX = int(cx - self.width / 2)
        startY = int(cy - self.height / 2)
        endX = int(cx + self.width / 2)
        endY = int(cy + self.height / 2)
        return (startX, startY, endX, endY)

################################################################################
# POSE ESTIMATION FUNCTIONS
################################################################################

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose, track_poses
from val import normalize, pad_width

def load_pose_model(ckpt_path, fp16=True):
    """Load the MobileNet-based pose model."""
    model = PoseEstimationWithMobileNet()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    load_state(model, checkpoint)
    model.eval().cuda()
    if fp16:
        model.half()
    return model

def infer_pose_fast(net, img, net_input_height_size,
                    stride=8, upsample_ratio=POSE_UPSAMPLE_RATIO, 
                    fp16=True,
                    pad_value=(0, 0, 0),
                    img_mean=np.array([128, 128, 128], np.float32),
                    img_scale=np.float32(1/256)):
    """Fast pose inference with optional FP16."""
    h, w, _ = img.shape
    scale = net_input_height_size / h
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0)
    if fp16:
        tensor_img = tensor_img.half()
    tensor_img = tensor_img.cuda()
    with torch.no_grad():
        stages_output = net(tensor_img)
    st2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(st2_heatmaps.squeeze().cpu().float().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    st2_pafs = stages_output[-1]
    pafs = np.transpose(st2_pafs.squeeze().cpu().float().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    return heatmaps, pafs, scale, pad

################################################################################
# MAIN PIPELINE
################################################################################

def main():
    # ---------------- CAPTURE PIPELINE (USB camera, v4l2src) ----------------
    cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: cannot open camera via v4l2src. Check USB camera connection.")
        return

    # Validate camera resolution and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {width}x{height} at {fps} FPS")

    # ---------------- STREAMING PIPELINE (UDP, H.264, MPEG-TS) ----------------
    video_writer = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, TARGET_FPS, (CAM_WIDTH, CAM_HEIGHT))
    if not video_writer.isOpened():
        print("Error: Could not open GStreamer video writer. Check your output pipeline.")
        return

    # Create face trackers
    trackers = []
    next_tracker_id = 0

    # Initialize face detection net
    if FACE_DETECTOR_ENABLED:
        face_net = cv2.dnn.readNetFromCaffe(FACE_CONFIG_FILE, FACE_MODEL_FILE)
        face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Load pose model if enabled
    if POSE_ESTIMATOR_ENABLED:
        pose_net = load_pose_model(POSE_CHECKPOINT_PATH, fp16=POSE_FP16)

    frame_count = 0
    prev_time = time.time()

    print("Starting combined pipeline.. Press Ctrl+C to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame!")
                break

            h, w = frame.shape[:2]

            #----------------- FACE DETECTION + ANONYMIZATION ---------------------
            if FACE_DETECTOR_ENABLED:
                # Advance trackers
                for trk in trackers:
                    trk.advance()

                # Face detection
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                face_detections = []
                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf > FACE_CONFIDENCE_THRESHOLD:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        bbox = box.astype(int)
                        face_detections.append((bbox, conf))

                assigned_ids = set()
                for (dbbox, dconf) in face_detections:
                    best_trk = None
                    best_iou = 0.0
                    for trk in trackers:
                        iou_val = iou(dbbox, trk.get_bbox())
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_trk = trk
                    if best_trk is not None and best_iou >= IOU_THRESHOLD:
                        best_trk.update(dbbox)
                        assigned_ids.add(best_trk.id)
                    else:
                        new_trk = FaceTracker(dbbox, next_tracker_id)
                        trackers.append(new_trk)
                        assigned_ids.add(new_trk.id)
                        next_tracker_id += 1

                for trk in trackers:
                    if trk.id not in assigned_ids:
                        trk.mark_missed()
                trackers = [t for t in trackers if t.time_since_update <= MAX_MISSED_FRAMES]

                # Anonymize detected faces
                for trk in trackers:
                    bb = trk.get_bbox()
                    sx, sy, ex, ey = bb
                    sx = max(0, sx); sy = max(0, sy)
                    ex = min(w, ex); ey = min(h, ey)
                    if trk.time_since_update > 0 and KALMAN_ENABLED:
                        scale_factor = RED_BOX_SCALE
                        cx, cy = trk.predicted_center
                        exp_w = trk.width * scale_factor
                        exp_h = trk.height * scale_factor
                        sx = int(cx - exp_w / 2)
                        sy = int(cy - exp_h / 2)
                        ex = int(cx + exp_w / 2)
                        ey = int(cy + exp_h / 2)
                        fill_color = (0, 0, 255)  # Red box for uncertain detection
                    else:
                        fill_color = (0, 0, 0)    # Black box for confirmed face
                    sx = max(0, sx); sy = max(0, sy)
                    ex = min(w, ex); ey = min(h, ey)
                    frame[sy:ey, sx:ex] = fill_color
                    # Optional ID text
                    cv2.putText(frame, f"ID:{trk.id}({trk.time_since_update})",
                                (sx, max(sy - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fill_color, 2)

            #----------------- POSE ESTIMATION ---------------------
            if POSE_ESTIMATOR_ENABLED:
                heatmaps, pafs, scale, pad = infer_pose_fast(
                    pose_net, frame, POSE_HEIGHT_SIZE,
                    stride=8, upsample_ratio=POSE_UPSAMPLE_RATIO,
                    fp16=POSE_FP16
                )
                num_kpts = Pose.num_kpts
                all_kpts_by_type = []
                tot_kpts_num = 0
                for kid in range(num_kpts):
                    tot_kpts_num += extract_keypoints(
                        heatmaps[:, :, kid],
                        all_kpts_by_type,
                        tot_kpts_num
                    )
                pose_entries, all_kpts = group_keypoints(all_kpts_by_type, pafs)
                for k_id in range(all_kpts.shape[0]):
                    all_kpts[k_id, 0] = (all_kpts[k_id, 0] * 8 / POSE_UPSAMPLE_RATIO - pad[1]) / scale
                    all_kpts[k_id, 1] = (all_kpts[k_id, 1] * 8 / POSE_UPSAMPLE_RATIO - pad[0]) / scale

                new_poses = []
                for n in range(len(pose_entries)):
                    if len(pose_entries[n]) == 0:
                        continue
                    pose_conf = pose_entries[n][18]  # overall confidence
                    if pose_conf < POSE_CONFIDENCE_THRESHOLD:
                        continue
                    kpts_arr = np.ones((num_kpts, 2), dtype=np.int32) * -1
                    for kid in range(num_kpts):
                        if pose_entries[n][kid] != -1.0:
                            idx = int(pose_entries[n][kid])
                            kpts_arr[kid, 0] = int(all_kpts[idx, 0])
                            kpts_arr[kid, 1] = int(all_kpts[idx, 1])
                    new_pose = Pose(kpts_arr, pose_conf)
                    new_poses.append(new_pose)

                if TRACK_POSE:
                    track_poses([], new_poses, threshold=3, smooth=SMOOTH_POSE)

                # Draw poses
                for p in new_poses:
                    p.draw(frame)
                    if POSE_FACE_ANONIMYZE:
                        HEAD_KPTS = [0, 14, 15, 16, 17]
                        valid_points = []
                        for kpt_id in HEAD_KPTS:
                            x, y = p.keypoints[kpt_id]
                            if x >= 0 and y >= 0:
                                valid_points.append((x, y))
                        if len(valid_points) > 0:
                            xs = [pt[0] for pt in valid_points]
                            ys = [pt[1] for pt in valid_points]
                            center_x = int(sum(xs) / len(xs))
                            center_y = int(sum(ys) / len(ys))
                            if len(valid_points) == 1:
                                radius = 40
                            else:
                                max_dist = 0
                                for (vx, vy) in valid_points:
                                    dist = ((vx - center_x) ** 2 + (vy - center_y) ** 2) ** 0.5
                                    if dist > max_dist:
                                        max_dist = dist
                                radius = int(max_dist + 20)
                            cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), -1)

            #-------------- FPS DISPLAY --------------
            frame_count += 1
            if frame_count % 10 == 0:
                cur_time = time.time()
                fps_now = frame_count / (cur_time - prev_time)
                prev_time = cur_time
                frame_count = 0
                if SHOW_FPS:
                    cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # ------------ Send the processed frame to the UDP pipeline -----------
            video_writer.write(frame)

    except KeyboardInterrupt:
        print("Exiting..")
    finally:
        cap.release()
        video_writer.release()

if __name__ == "__main__":
    main()
