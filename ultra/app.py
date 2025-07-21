from flask import Flask, render_template, request, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO("yolov8x-worldv2.pt")  # 경로 확인 필수

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 웹캠 기본 번호

@app.route('/')
def index():
    return render_template('index.html')  # HTML 페이지 렌더링

def generate_frames(target_classes):
    if not cap.isOpened():
        return "Error: Unable to open webcam."

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # YOLO 탐지 수행
        results = model.predict(frame)
        detections = [
            det for det in results[0].boxes.data
            if det[5] in target_classes
        ]

        # 결과 프레임 생성
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = model.names[int(cls)]
            if label in target_classes:
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # JPEG 변환
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    target_classes = request.form.get('classes', '').split(',')
    target_classes = [cls.strip() for cls in target_classes if cls.strip()]
    if not target_classes:
        return "Error: No classes provided."

    return Response(generate_frames(target_classes), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
