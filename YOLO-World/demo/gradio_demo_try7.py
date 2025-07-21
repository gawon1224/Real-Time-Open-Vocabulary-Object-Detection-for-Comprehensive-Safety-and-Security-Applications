# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from functools import partial
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.dataset import Compose
from torchvision.ops import nms
import time  # 실시간 프레임 간 딜레이를 추가하기 위해 사용

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='Path to the config file')
    parser.add_argument('checkpoint', help='Path to the model checkpoint')
    parser.add_argument(
        '--work-dir',
        help='Directory to save the evaluation results',
        default='./work_dirs'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        help='Override config options. Format: key=value'
    )
    return parser.parse_args()

def run_image(image, text, max_num_boxes, score_thr, nms_thr):
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    image_np = np.array(image, dtype=np.uint8)
    data_info = dict(img_id=0, img=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    for bbox, label, score in zip(pred_instances['bboxes'], pred_instances['labels'], pred_instances['scores']):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{texts[label][0]}: {score:.2f}"
        cv2.putText(image_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

def webcam_real_time(text, max_num_boxes, score_thr, nms_thr):
    """
    Capture frames from the webcam, process them, and return in real-time.
    """
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Error: Webcam could not be opened")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Convert BGR to RGB and process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        annotated_image = run_image(pil_image, text, max_num_boxes, score_thr, nms_thr)

        # Yield the processed frame
        yield np.array(annotated_image)
        time.sleep(0.03)  # 약간의 딜레이 추가

    camera.release()

def create_gradio_interface():
    """
    Create Gradio interface for YOLO-World real-time detection.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# YOLO-World: Real-Time Object Detection")

        with gr.Row():
            input_text = gr.Textbox(label="Classes to detect (comma-separated)", value="car, person")
            max_num_boxes = gr.Slider(label="Maximum Number of Boxes", minimum=1, maximum=100, value=10, step=1)
            score_thr = gr.Slider(label="Score Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
            nms_thr = gr.Slider(label="NMS Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.01)

        with gr.Row():
            webcam_button = gr.Button("Start Real-Time Detection")
            output_image = gr.Image(type="numpy", label="Detection Output")

        # 실시간 업데이트를 위해 Gradio 업데이트 함수 사용
        def real_time_update(*args):
            for frame in webcam_real_time(*args):
                yield gr.update(value=frame)

        webcam_button.click(
            real_time_update,
            inputs=[input_text, max_num_boxes, score_thr, nms_thr],
            outputs=output_image,
        )

        demo.launch(server_name="0.0.0.0", server_port=8080)

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir if args.work_dir else os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    cfg.load_from = args.checkpoint

    global runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    create_gradio_interface()
