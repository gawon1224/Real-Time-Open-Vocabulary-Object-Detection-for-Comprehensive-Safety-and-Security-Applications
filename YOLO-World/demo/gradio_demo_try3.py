# Copyright (c) Tencent Inc. All rights reserved.
import os
import sys
import argparse
import os.path as osp
from io import BytesIO
from functools import partial

import cv2
import onnx
import torch
import onnxsim
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from torchvision.ops import nms
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.config import Config, DictAction, ConfigDict
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

import sys
sys.path.append('/home/device03/YOLO-World/deploy')
from easydeploy import model as EM

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics',
        default='output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def run_image(runner,
              image=None,
              text="",
              max_num_boxes=100,
              score_thr=0.05,
              nms_thr=0.7,
              is_webcam=False):
    """
    Run detection on an input image or stream from webcam.
    Args:
        runner: The model runner.
        image: Input image (if not using webcam).
        text: Class labels for detection.
        max_num_boxes: Maximum number of boxes to display.
        score_thr: Score threshold for detection.
        nms_thr: NMS threshold for detection.
        is_webcam: Flag to indicate if input is from webcam.
    """
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]

    if is_webcam:
        # 웹캠 처리 로직
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            sys.exit()

        while True:
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # PIL 이미지를 NumPy 배열로 변환
            image_np = np.array(pil_image, dtype=np.uint8)
            data_info = dict(img_id=0, img=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), texts=texts)
            data_info = runner.pipeline(data_info)
            data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                              data_samples=[data_info['data_samples']])

            with autocast(enabled=False), torch.no_grad():
                output = runner.model.test_step(data_batch)[0]
                pred_instances = output.pred_instances

            keep = nms(pred_instances.bboxes,
                       pred_instances.scores,
                       iou_threshold=nms_thr)
            pred_instances = pred_instances[keep]
            pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

            if len(pred_instances.scores) > max_num_boxes:
                indices = pred_instances.scores.float().topk(max_num_boxes)[1]
                pred_instances = pred_instances[indices]

            pred_instances = pred_instances.cpu().numpy()
            if 'masks' in pred_instances:
                masks = pred_instances['masks']
            else:
                masks = None
            detections = sv.Detections(xyxy=pred_instances['bboxes'],
                                       class_id=pred_instances['labels'],
                                       confidence=pred_instances['scores'],
                                       mask=masks)

            labels = [
                f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
                zip(detections.class_id, detections.confidence)
            ]

            frame = BOUNDING_BOX_ANNOTATOR.annotate(frame, detections)
            frame = LABEL_ANNOTATOR.annotate(frame, detections, labels=labels)
            if masks is not None:
                frame = MASK_ANNOTATOR.annotate(frame, detections)

            cv2.imshow("YOLO-World Real-Time Detection", frame)

            # 'q' 키를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()
    else:
        # 이미지 처리 로직
        image_np = np.array(image, dtype=np.uint8)
        data_info = dict(img_id=0, img=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), texts=texts)
        data_info = runner.pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])

        with autocast(enabled=False), torch.no_grad():
            output = runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        keep = nms(pred_instances.bboxes,
                   pred_instances.scores,
                   iou_threshold=nms_thr)
        pred_instances = pred_instances[keep]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        if 'masks' in pred_instances:
            masks = pred_instances['masks']
        else:
            masks = None
        detections = sv.Detections(xyxy=pred_instances['bboxes'],
                                   class_id=pred_instances['labels'],
                                   confidence=pred_instances['scores'],
                                   mask=masks)

        labels = [
            f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
            zip(detections.class_id, detections.confidence)
        ]

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        if masks is not None:
            image = MASK_ANNOTATOR.annotate(image, detections)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image



def capture_and_detect(runner, input_text, max_num_boxes, score_thr, nms_thr):
    camera = cv2.VideoCapture(0)  # 웹캠 연결
    if not camera.isOpened():
        print("Error: Could not open camera")
        sys.exit()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        annotated_image = run_image(
            runner=runner,
            image=pil_image,
            text=input_text,
            max_num_boxes=max_num_boxes,
            score_thr=score_thr,
            nms_thr=nms_thr
        )

        annotated_frame = np.array(annotated_image)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLO-World Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


def demo(runner, args):
    with gr.Blocks(title="YOLO-World") as demo:
        with gr.Row():
            gr.Markdown("""
                <h1 style="text-align: center;">Intelligent Robot Project (UOS AI)</h1>
                <h2 style="text-align: center; margin-top: 10px;">YOLO-World: Real-Time Open-Vocabulary</h2>
            """)
        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    image = gr.Image(type='pil', label='Input Image')
                input_text = gr.Textbox(
                    lines=7,
                    label='Enter the classes to be detected, '
                          'separated by comma',
                    value=', '.join(CocoDataset.METAINFO['classes']),
                    elem_id='textbox')

                with gr.Row():
                    submit = gr.Button('Submit')
                    clear = gr.Button('Clear')
                    webcam = gr.Button('Start Webcam Detection')

                with gr.Row():
                    export = gr.Button('Deploy and Export ONNX Model')
                out_download = gr.File(visible=False)
                max_num_boxes = gr.Slider(minimum=1,
                                          maximum=300,
                                          value=100,
                                          step=1,
                                          interactive=True,
                                          label='Maximum Number Boxes')
                score_thr = gr.Slider(minimum=0,
                                      maximum=1,
                                      value=0.05,
                                      step=0.001,
                                      interactive=True,
                                      label='Score Threshold')
                nms_thr = gr.Slider(minimum=0,
                                    maximum=1,
                                    value=0.7,
                                    step=0.001,
                                    interactive=True,
                                    label='NMS Threshold')
            with gr.Column(scale=0.7):
                output_image = gr.Image(type='pil', label='Output Image')

        submit.click(partial(run_image, runner),
                     [image, input_text, max_num_boxes, score_thr, nms_thr],
                     [output_image])
        clear.click(lambda: [None, '', None], None,
                    [image, input_text, output_image])
        webcam.click(partial(capture_and_detect, runner),
                     [input_text, max_num_boxes, score_thr, nms_thr],
                     None)

        demo.launch(server_name='0.0.0.0',
                    server_port=8080)


if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

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
    demo(runner, args)
