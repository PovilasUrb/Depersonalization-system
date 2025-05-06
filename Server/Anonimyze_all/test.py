import yaml
import cv2
import time
import numpy as np
import torch
import threading
import subprocess

################################################################################
# 1) Load YAML config
################################################################################
def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

################################################################################
# 2) Face Detection & Tracking Helpers
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
    def __init__(self, bbox, tracker_id, kalman_enabled=True):
        self.id = tracker_id
        self.last_bbox = bbox  # (startX, startY, endX, endY)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.time_since_update = 0
        self.kalman_enabled = kalman_enabled

        if self.kalman_enabled:
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
        if self.kalman_enabled:
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

        if self.kalman_enabled:
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
# 3) Pose Estimation (Lightweight OpenPose) on XPU
################################################################################
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose, track_poses
from val import normalize, pad_width

def load_pose_model(ckpt_path, fp16=True):
    """Load the MobileNet-based pose model and move it to XPU."""
    fp16=False
    model = PoseEstimationWithMobileNet()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    load_state(model, checkpoint)
    model.eval()

    # 1) Move to XPU if available
    # Make sure you installed PyTorch with XPU support
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model.to(device)

    # 2) Optionally half precision
    if fp16:
        model.half()

    return model

def infer_pose_fast(net, img, net_input_height_size,
                    stride=8, upsample_ratio=1, 
                    fp16=True,
                    pad_value=(0, 0, 0),
                    img_mean=np.array([128, 128, 128], np.float32),
                    img_scale=np.float32(1/256)):
    """Fast pose inference on XPU (if available)."""
    device = next(net.parameters()).device

    h, w, _ = img.shape
    scale = net_input_height_size / h
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0)

    if fp16:
        tensor_img = tensor_img.half()

    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        stages_output = net(tensor_img)

    st2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(st2_heatmaps.squeeze().float().cpu().numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    st2_pafs = stages_output[-1]
    pafs = np.transpose(st2_pafs.squeeze().float().cpu().numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

################################################################################
# 4) process_stream: read RTSP -> anonymize -> pipe to FFmpeg -> mediamtx
################################################################################
def process_stream(stream_config, global_config):
    """
    Steps per stream:
      1) Read from RTSP input using OpenCV + FFmpeg.
      2) Face + Pose anonymization (Pose on XPU).
      3) Pipe frames to an FFmpeg subprocess that pushes to mediamtx.
    """
    # ------------------ Parse relevant config -------------------
    face_detector_enabled = stream_config.get("face_detector_enabled", 
                                global_config["face_detector_enabled"])
    pose_estimator_enabled = stream_config.get("pose_estimator_enabled",
                                global_config["pose_estimator_enabled"])
    show_fps = global_config["show_fps"]

    face_model_file = global_config["face_model_file"]
    face_config_file = global_config["face_config_file"]
    face_conf_thresh = global_config["face_confidence_threshold"]

    kalman_enabled = global_config["kalman_enabled"]
    iou_threshold = global_config["iou_threshold"]
    max_missed_frames = global_config["max_missed_frames"]
    red_box_scale = global_config["red_box_scale"]

    pose_checkpoint_path = global_config["pose_checkpoint_path"]
    pose_height_size = global_config["pose_height_size"]
    pose_upsample_ratio = global_config["pose_upsample_ratio"]
    pose_fp16 = global_config["pose_fp16"]
    pose_conf_threshold = global_config["pose_confidence_threshold"]
    track_pose_flag = global_config["track_pose"]
    smooth_pose_flag = global_config["smooth_pose"]
    pose_face_anonymize = global_config["pose_face_anonymize"]

    input_url = stream_config["input_url"]
    output_url = stream_config["output_url"]

    # "Fallback" or desired camera dims / fps if the RTSP doesn't provide them
    cam_width = global_config["cam_width"]
    cam_height = global_config["cam_height"]
    target_fps = float(global_config["target_fps"])

    cap = cv2.VideoCapture(input_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[{stream_config['name']}] Error: cannot open input URL {input_url}.")
        return

    # Query actual properties from the incoming RTSP
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cam_width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cam_height
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = target_fps

    print(f"[{stream_config['name']}] Capturing {width}x{height} @ {fps:.1f} FPS from {input_url}")

    # Initialize face net if needed
    face_net = None
    if face_detector_enabled:
        face_net = cv2.dnn.readNetFromCaffe(face_config_file, face_model_file)
        #
        # Optional: If you want to attempt using OpenCL (which might run on Intel Arc),
        # uncomment these lines. YMMV with actual hardware support:
        #
        # face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCL)
        # face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    # Initialize pose net (XPU)
    pose_net = None
    if pose_estimator_enabled:
        pose_net = load_pose_model(pose_checkpoint_path, fp16=pose_fp16)

    trackers = []
    next_tracker_id = 0

    # Start an FFmpeg process to push frames to mediamtx at output_url
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                          # overwrite output if any
        "-f", "rawvideo",             # input format from stdin is raw video
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",                    # stdin
        "-c:v", "libx264",            # H.264 encoder
        "-preset", "ultrafast",
        "-f", "rtsp",                 # RTSP output format
        output_url
    ]
    print(f"[{stream_config['name']}] Launching FFmpeg to RTSP => {output_url}")
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print(f"[{stream_config['name']}] Anonymization pipeline started.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[{stream_config['name']}] Stream ended or read error.")
                break

            # 1) FACE DETECTION + TRACKING
            if face_detector_enabled and face_net is not None:
                for trk in trackers:
                    trk.advance()

                h_frame, w_frame = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                             (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                face_detections = []
                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf > face_conf_thresh:
                        box = detections[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
                        bbox = box.astype(int)
                        face_detections.append((bbox, conf))

                assigned_ids = set()
                for (dbbox, dconf) in face_detections:
                    best_trk = None
                    best_iou_val = 0.0
                    for trk in trackers:
                        iou_val = iou(dbbox, trk.get_bbox())
                        if iou_val > best_iou_val:
                            best_iou_val = iou_val
                            best_trk = trk
                    if best_trk is not None and best_iou_val >= iou_threshold:
                        best_trk.update(dbbox)
                        assigned_ids.add(best_trk.id)
                    else:
                        new_trk = FaceTracker(dbbox, next_tracker_id, kalman_enabled=kalman_enabled)
                        trackers.append(new_trk)
                        assigned_ids.add(new_trk.id)
                        next_tracker_id += 1

                # Remove old trackers
                for trk in trackers:
                    if trk.id not in assigned_ids:
                        trk.mark_missed()
                trackers = [t for t in trackers if t.time_since_update <= max_missed_frames]

                # Anonymize faces
                for trk in trackers:
                    sx, sy, ex, ey = trk.get_bbox()
                    sx = max(0, sx); sy = max(0, sy)
                    ex = min(w_frame, ex); ey = min(h_frame, ey)

                    if trk.time_since_update > 0 and kalman_enabled:
                        scale_factor = red_box_scale
                        cx, cy = trk.predicted_center
                        exp_w = trk.width * scale_factor
                        exp_h = trk.height * scale_factor
                        sx = int(cx - exp_w / 2)
                        sy = int(cy - exp_h / 2)
                        ex = int(cx + exp_w / 2)
                        ey = int(cy + exp_h / 2)
                        # clamp again
                        sx = max(0, sx); sy = max(0, sy)
                        ex = min(w_frame, ex); ey = min(h_frame, ey)

                    frame[sy:ey, sx:ex] = (0, 0, 0)  # black box

            # 2) POSE ESTIMATION (ON XPU) + HEAD ANON
            if pose_estimator_enabled and pose_net is not None:
                heatmaps, pafs, scale, pad = infer_pose_fast(
                    pose_net, frame, pose_height_size,
                    stride=8, upsample_ratio=pose_upsample_ratio,
                    fp16=pose_fp16
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
                    all_kpts[k_id, 0] = (all_kpts[k_id, 0] * 8 / pose_upsample_ratio - pad[1]) / scale
                    all_kpts[k_id, 1] = (all_kpts[k_id, 1] * 8 / pose_upsample_ratio - pad[0]) / scale

                new_poses = []
                for n in range(len(pose_entries)):
                    if len(pose_entries[n]) == 0:
                        continue
                    pose_conf = pose_entries[n][18]  # overall confidence
                    if pose_conf < pose_conf_threshold:
                        continue
                    kpts_arr = np.ones((num_kpts, 2), dtype=np.int32) * -1
                    for kid in range(num_kpts):
                        if pose_entries[n][kid] != -1.0:
                            idx = int(pose_entries[n][kid])
                            kpts_arr[kid, 0] = int(all_kpts[idx, 0])
                            kpts_arr[kid, 1] = int(all_kpts[idx, 1])
                    new_pose = Pose(kpts_arr, pose_conf)
                    new_poses.append(new_pose)

                if track_pose_flag:
                    track_poses([], new_poses, threshold=3, smooth=smooth_pose_flag)

                # Example: draw skeletons
                for p in new_poses:
                    p.draw(frame)

                    if pose_face_anonymize:
                        HEAD_KPTS = [0, 14, 15, 16, 17]
                        valid_points = []
                        for kpt_id in HEAD_KPTS:
                            x, y = p.keypoints[kpt_id]
                            if x >= 0 and y >= 0:
                                valid_points.append((x, y))
                        if valid_points:
                            xs = [pt[0] for pt in valid_points]
                            ys = [pt[1] for pt in valid_points]
                            center_x = int(sum(xs) / len(xs))
                            center_y = int(sum(ys) / len(ys))
                            if len(valid_points) == 1:
                                radius = 40
                            else:
                                max_dist = max(
                                    ((vx - center_x)**2 + (vy - center_y)**2)**0.5
                                    for (vx, vy) in valid_points
                                )
                                radius = int(max_dist + 20)
                            cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), -1)

            # 3) Pipe anonymized frame to FFmpeg
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print(f"[{stream_config['name']}] FFmpeg pipe closed unexpectedly.")
                break

            # 4) Show optional FPS
            if show_fps:
                if not hasattr(process_stream, 'last_time'):
                    process_stream.last_time = time.perf_counter()
                else:
                    now = time.perf_counter()
                    fps_now = 1.0 / (now - process_stream.last_time) if now != process_stream.last_time else 0
                    process_stream.last_time = now
                    print(f"[{stream_config['name']}] FPS: {fps_now:.2f}")

    except KeyboardInterrupt:
        print(f"[{stream_config['name']}] KeyboardInterrupt, stopping..")

    finally:
        cap.release()
        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"[{stream_config['name']}] Stopped.")

################################################################################
# 5) Main
################################################################################
def main():
    config = load_config("config.yaml")
    global_config = config["global_config"]
    streams = config["streams"]

    threads = []
    for s in streams:
        t = threading.Thread(target=process_stream, args=(s, global_config), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Main thread: KeyboardInterrupt, exiting...")

if __name__ == "__main__":
    main()
