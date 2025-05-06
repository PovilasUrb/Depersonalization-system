import cv2
import subprocess
import numpy as np

def main():
    # Input RTSP URL from your source (e.g., a USB camera on an RTSP server)
    input_url = "rtsp://127.0.0.1:8554/camera1"
    # Output RTSP URL for mediamtx; adjust the endpoint and port as needed
    output_url = "rtsp://127.0.0.1:8554/live/camera1_anon"

    # Open the input stream using FFmpeg backend
    cap = cv2.VideoCapture(input_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream.")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 25.0  # Fallback FPS if the stream doesn't provide it

    print(f"Input stream opened: {width}x{height} @ {fps} fps")

    # Create the FFmpeg command that will read raw BGR frames from stdin and push an H.264 encoded RTSP stream
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                                  # Overwrite output files without asking
        "-f", "rawvideo",                      # Input is raw video
        "-pix_fmt", "bgr24",                   # Pixel format (OpenCV frames are in bgr24)
        "-s", f"{width}x{height}",              # Frame size
        "-r", str(fps),                        # Frame rate
        "-i", "-",                             # Input comes from stdin
        "-c:v", "libx264",                     # Encode video with H.264
        "-preset", "ultrafast",                # Preset for low latency (adjust if needed)
        "-f", "rtsp",                          # Output format is RTSP
        output_url
    ]

    # Start the FFmpeg process with a pipe for stdin
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    print("FFmpeg process started, streaming to mediamtx endpoint:", output_url)

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Stream ended or error encountered.")
            break

        # --- Anonymization step: draw a black rectangle in the center ---
        h_frame, w_frame = frame.shape[:2]
        cv2.rectangle(frame,
                      (w_frame // 4, h_frame // 4),
                      (3 * w_frame // 4, 3 * h_frame // 4),
                      (0, 0, 0),
                      thickness=-1)

        # Write the processed frame to FFmpeg's stdin as raw bytes.
        try:
            process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg pipe closed, stopping the stream.")
            break

        # Optionally, show a local preview.
        cv2.imshow("Preview", frame)
        # Exit on 'Esc' key
        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting on user request.")
            break

    # Clean up: release the capture, close the pipe, and wait for FFmpeg to finish.
    cap.release()
    if process.stdin:
        process.stdin.close()
    process.wait()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()