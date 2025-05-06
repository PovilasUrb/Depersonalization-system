import cv2
import subprocess
import time

width = 640
height = 360
fps = 25

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

ffmpeg = subprocess.Popen([
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',  # Input from stdin
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-b:v', '1M',    # Increase video bitrate
    '-f', 'rtsp',
    'rtsp://localhost:8554/raw'
], stdin=subprocess.PIPE)

print("Streaming to rtsp://localhost:8554/raw ...")

frame_interval = 1.0 / fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break
    # Force resize the frame
    frame = cv2.resize(frame, (width, height))
    
    try:
        ffmpeg.stdin.write(frame.tobytes())
        ffmpeg.stdin.flush()
    except BrokenPipeError:
        print("FFmpeg pipe broken")
        break

    elapsed = time.time() - start_time
    sleep_time = frame_interval - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
