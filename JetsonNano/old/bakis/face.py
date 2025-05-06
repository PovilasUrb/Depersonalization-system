import cv2
import time
import subprocess
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 15
RTSP_URL = "rtsp://localhost:8554/live"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5

# Tracking parameters
IOU_THRESHOLD = 0.3        # Minimum IoU to associate detection with tracker
MAX_MISSED_FRAMES = 5      # Remove tracker if missed more than this many frames

# Red bounding box scale factor (for trackers that missed an update)
RED_BOX_SCALE = 1.4

# ---------------------------
# FFmpeg command for RTSP streaming
# ---------------------------
ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "{}x{}".format(CAM_WIDTH, CAM_HEIGHT),
    "-r", str(TARGET_FPS),
    "-i", "-",  # read input from stdin
    "-rtsp_transport", "tcp",  # force TCP transport
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    RTSP_URL
]

ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# ---------------------------
# Utility: Intersection over Union (IoU)
# ---------------------------
def iou(bbox1, bbox2):
    # bbox format: (startX, startY, endX, endY)
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

# ---------------------------
# FaceTracker Class using Kalman Filter
# ---------------------------
class FaceTracker:
    def __init__(self, bbox, tracker_id):
        self.id = tracker_id
        self.last_bbox = bbox  # (startX, startY, endX, endY)
        # Calculate center and size.
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        # Initialize Kalman filter: state = [cx, cy, vx, vy] as a column vector.
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        init_state = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePre = init_state.copy()
        self.kalman.statePost = init_state.copy()
        self.predicted_center = (cx, cy)
        self.time_since_update = 0

    def advance(self):
        # Predict once per frame.
        pred = self.kalman.predict()
        self.predicted_center = (pred[0, 0], pred[1, 0])

    def update(self, bbox):
        self.last_bbox = bbox
        self.time_since_update = 0
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measurement)
        self.predicted_center = (cx, cy)

    def mark_missed(self):
        self.time_since_update += 1

    def get_bbox(self):
        # Compute bounding box from predicted center.
        cx, cy = self.predicted_center
        startX = int(cx - self.width / 2)
        startY = int(cy - self.height / 2)
        endX = int(cx + self.width / 2)
        endY = int(cy + self.height / 2)
        return (startX, startY, endX, endY)

# ---------------------------
# List for Active Trackers
# ---------------------------
trackers = []
next_tracker_id = 0

# ---------------------------
# Load the Face Detector Model (SSD Res10)
# ---------------------------
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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

prev_time = time.time()
frame_count = 0

print("Starting face detection and tracking pipeline... Press Ctrl+C to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame!")
        break

    h, w = frame.shape[:2]

    # Advance all trackers once per frame.
    for tracker in trackers:
        tracker.advance()

    # ---------------------------
    # Run Face Detection on the Frame
    # ---------------------------
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Build list of detections: each is (bbox, confidence)
    detections_list = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            bbox = box.astype("int")
            detections_list.append((bbox, confidence))

    # ---------------------------
    # Associate Detections with Existing Trackers Using IoU
    # ---------------------------
    assigned_tracker_ids = set()
    for det_bbox, det_conf in detections_list:
        best_tracker = None
        best_iou = 0
        for tracker in trackers:
            tracker_bbox = tracker.get_bbox()
            current_iou = iou(det_bbox, tracker_bbox)
            if current_iou > best_iou:
                best_iou = current_iou
                best_tracker = tracker
        if best_tracker is not None and best_iou >= IOU_THRESHOLD:
            best_tracker.update(det_bbox)
            assigned_tracker_ids.add(best_tracker.id)
        else:
            # Create a new tracker if no match is found.
            new_tracker = FaceTracker(det_bbox, next_tracker_id)
            trackers.append(new_tracker)
            assigned_tracker_ids.add(new_tracker.id)
            next_tracker_id += 1

    # Mark trackers that did not get updated.
    for tracker in trackers:
        if tracker.id not in assigned_tracker_ids:
            tracker.mark_missed()

    # Remove trackers that have missed too many frames.
    trackers = [trk for trk in trackers if trk.time_since_update <= MAX_MISSED_FRAMES]

    # ---------------------------
    # Draw Tracker Bounding Boxes (Anonymization)
    # ---------------------------
    for tracker in trackers:
        # Get predicted bbox.
        bbox = tracker.get_bbox()
        startX, startY, endX, endY = bbox
        # Clamp bbox to frame boundaries.
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        if tracker.time_since_update > 0:
            # If missed update, enlarge the bbox.
            scale = RED_BOX_SCALE
            cx, cy = tracker.predicted_center
            exp_width = tracker.width * scale
            exp_height = tracker.height * scale
            startX = int(cx - exp_width / 2)
            startY = int(cy - exp_height / 2)
            endX = int(cx + exp_width / 2)
            endY = int(cy + exp_height / 2)
            fill_color = (0, 0, 255)  # red for missed detection
        else:
            fill_color = (0, 0, 0)    # black for updated detection
        # Draw the filled rectangle.
        frame[startY:endY, startX:endX] = fill_color
        # Optionally, overlay tracker ID and missed count.
        cv2.putText(frame, f"ID: {tracker.id} ({tracker.time_since_update})",
                    (startX, max(startY - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fill_color, 2)

    # ---------------------------
    # Overlay FPS on the Frame
    # ---------------------------
    frame_count += 1
    if frame_count % 10 == 0:
        current_time = time.time()
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ---------------------------
    # Stream the Annotated Frame via FFmpeg (RTSP)
    # ---------------------------
    try:
        ffmpeg_process.stdin.write(frame.tobytes())
    except Exception as e:
        print("FFmpeg error:", e)
        break

cap.release()
if ffmpeg_process.stdin:
    ffmpeg_process.stdin.close()
ffmpeg_process.wait()
