import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# YOLO 모델 초기화
model = YOLO("yolov8s-world.pt")  # 모델 파일 경로

# 사용자 지정 클래스 설정 (필요시)
model.set_classes(["chair", "desk", "monitor", "man", "cup", "mouse"])

# 웹캠 설정 (특정 인덱스)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError(f"Unable to access webcam with index 1")


# YOLO 감지 함수
def yolo_detection():
    """
    OpenCV에서 특정 웹캠을 사용하여 프레임을 가져오고 YOLO 감지를 수행합니다.
    """
    success, frame = cap.read()
    if not success:
        return None

    # YOLO 예측 수행
    results = model.predict(frame)

    # 감지된 객체가 주석된 이미지 생성
    annotated_frame = results[0].plot()

    # OpenCV 이미지를 다시 RGB로 변환 (Gradio 출력용)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    return annotated_frame


# Gradio 인터페이스 구성
output = gr.Image(label="YOLO Detections")  # 처리 결과 출력

# Gradio 앱 구성
demo = gr.Interface(
    fn=yolo_detection,  # YOLO 감지 함수
    inputs=None,  # 입력 없음 (OpenCV로 직접 처리)
    outputs=output,  # 출력: 감지된 이미지
    live=True  # 실시간 활성화
)

if __name__ == "__main__":
    try:
        demo.launch(server_port=8080)  # 로컬 호스트 포트 번호 지정
    finally:
        cap.release()  # 앱 종료 시 웹캠 해제
