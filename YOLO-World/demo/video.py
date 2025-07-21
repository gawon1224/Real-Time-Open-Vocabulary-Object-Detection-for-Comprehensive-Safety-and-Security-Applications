import cv2

video_path = "/home/device03/YOLO-World/demo/sample_images/classroom.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Cannot open video file: {video_path}")
else:
    print("Video file opened successfully!")
