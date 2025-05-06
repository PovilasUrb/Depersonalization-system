import cv2
import torch
from mmpose.apis import init_pose_model, inference_top_down_pose_model

def main():
    # Paths for config and checkpoint (downloaded earlier)
    pose_config = "mobilenetv2_coco_256x192.py"
    pose_checkpoint = "mobilenetv2_coco_256x192.pth"
    
    # Set device: use XPU if available, else CPU
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print("Using device:", device)
    
    # Initialize the pose model with MMPose API
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    
    # RTSP input URL
    rtsp_url = "rtsp://127.0.0.1:8554/camera1"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream:", rtsp_url)
        return
    
    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
    print("Starting pose estimation. Press ESC to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Stream ended or error encountered.")
                break
            
            h, w = frame.shape[:2]
            # Use the entire frame as a bounding box for a single person
            person_bbox = [0, 0, w, h]  # [x1, y1, x2, y2]
            person_results = [{'bbox': person_bbox}]
            
            # Run MMPose top-down pose estimation
            pose_results, _ = inference_top_down_pose_model(
                pose_model, frame, person_results, bbox_format='xyxy'
            )
            
            # Draw keypoints on the frame
            for res in pose_results:
                keypoints = res['keypoints']  # shape: (N, 3) where each keypoint is (x, y, confidence)
                for (x, y, conf) in keypoints:
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
