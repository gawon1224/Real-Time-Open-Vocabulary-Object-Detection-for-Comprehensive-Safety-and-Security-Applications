# Copyright (c) Tencent Inc. All rights reserved.
import os
import sys
from functools import partial

import cv2
import numpy as np
import gradio as gr
from PIL import Image

# YOLO 디텍션 함수
def process_frame(frame, max_num_boxes, score_thr, nms_thr, text):
    """
    Run detection on a single frame (dummy implementation).
    Args:
        frame: Input image as a NumPy array.
        max_num_boxes: Maximum number of boxes to display.
        score_thr: Detection score threshold.
        nms_thr: Non-Maximum Suppression threshold.
        text: Classes to detect (comma-separated string).
    Returns:
        Processed image as a NumPy array with detections annotated.
    """
    # 더미 YOLO 디텍션 결과 그리기 (사용자 모델과 연결 가능)
    height, width, _ = frame.shape
    cv2.rectangle(frame, (50, 50), (width - 50, height - 50), (0, 255, 0), 2)
    label_text = f"Detected: {text} (Score: {score_thr})"
    cv2.putText(frame, label_text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# 이미지 디텍션 함수
def detect_from_image(image, max_num_boxes, score_thr, nms_thr, text):
    frame = np.array(image)
    result = process_frame(frame, max_num_boxes, score_thr, nms_thr, text)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# 웹캠 디텍션 함수
def detect_from_webcam(max_num_boxes, score_thr, nms_thr, text, camera_index=0):
    """
    Capture a frame from the webcam and perform detection.
    Args:
        max_num_boxes: Maximum number of boxes to display.
        score_thr: Detection score threshold.
        nms_thr: Non-Maximum Suppression threshold.
        text: Classes to detect (comma-separated string).
        camera_index: Index of the camera to use (default: 0).
    Returns:
        Processed image as a NumPy array with detections annotated.
    """
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        return "Error: Could not open camera"

    ret, frame = camera.read()
    camera.release()

    if not ret:
        return "Error: Failed to capture image"

    result = process_frame(frame, max_num_boxes, score_thr, nms_thr, text)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Gradio 인터페이스 정의
def create_gradio_interface():
    with gr.Blocks(title="YOLO-World Gradio Interface") as demo:
        gr.Markdown("# YOLO-World: Real-Time Detection with Gradio")

        # Input 영역
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload an Image")
            webcam_button = gr.Button("Capture from Webcam")

        # 설정 영역
        with gr.Row():
            input_text = gr.Textbox(
                lines=2, label="Classes to detect (comma-separated)", value="car, person"
            )
            max_num_boxes = gr.Slider(
                minimum=1, maximum=100, value=10, step=1, label="Maximum Number of Boxes"
            )
            score_thr = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Score Threshold"
            )
            nms_thr = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="NMS Threshold"
            )

        # Output 영역
        with gr.Row():
            output_image = gr.Image(type="numpy", label="Detection Output")

        # 버튼 동작 정의
        input_image.change(
            detect_from_image,
            inputs=[input_image, max_num_boxes, score_thr, nms_thr, input_text],
            outputs=output_image,
        )
        webcam_button.click(
            detect_from_webcam,
            inputs=[max_num_boxes, score_thr, nms_thr, input_text],
            outputs=output_image,
        )

        demo.launch(server_name="0.0.0.0", server_port=8080)

# Gradio 인터페이스 실행
if __name__ == "__main__":
    create_gradio_interface()
