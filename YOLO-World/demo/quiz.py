import random
import re
import cv2
import matplotlib.pyplot as plt
import gradio_demo_try1
import pandas as pd

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

def calculate_iou(bbox1, bbox2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    :param bbox1: [x_min, y_min, x_max, y_max] for the first bounding box
    :param bbox2: [x_min, y_min, x_max, y_max] for the second bounding box
    :return: IoU value (float)
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    
    # Calculate intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # If there is no intersection, IoU is 0
    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return 0.0

    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

    # Calculate union
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area
    return iou


def visualize_bounding_boxes(image, objects):
    # 이미지가 PIL 형식이라면 numpy 배열로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 이미지 복사 (원본 이미지를 수정하지 않기 위해)
    image_copy = image.copy()

    # 객체의 바운딩박스 시각화
    for obj in objects:
        # 바운딩 박스 정보와 클래스 이름 추출
        bbox = obj['bbox']
        class_name = obj['class_name'][0]
        
        # bbox 값이 float일 수 있으므로 int로 변환
        bbox = [int(coord) for coord in bbox]
        
        # 바운딩박스 그리기 (x1, y1, x2, y2)
        cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 클래스명 텍스트 표시
        # cv2.putText(image_copy, class_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 시각화
    plt.imshow(image_copy)
    plt.axis('off')
    plt.show()


def intersect_objects(previous_candidates, new_inferred_objects):
    # 객체 클래스 기준으로 교집합 계산
    common_objects = [
        obj for obj in new_inferred_objects if obj['class_name'][0] in {o['class_name'][0] for o in previous_candidates}
    ]
    return common_objects


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

import csv
def load_objects_info_from_csv(filename="objects_info.csv"):
    # 파일이 없으면 예시 데이터를 생성하여 저장
    if not os.path.exists(filename):
        # 예시 객체 정보 (임시로 생성한 데이터)
        objects_info = [
            {'class_id': 0, 'class_name': 'person', 'bbox': [100, 150, 200, 250], 'confidence': 0.98},
            {'class_id': 1, 'class_name': 'car', 'bbox': [300, 350, 500, 450], 'confidence': 0.95},
        ]
        # CSV 파일로 저장
        save_objects_info_to_csv(objects_info, filename)
    else:
        # 파일이 있으면 그대로 읽어옴
        df = pd.read_csv(filename)
        objects_info = df.to_dict(orient='records')
    
    return objects_info

def save_objects_info_to_csv(objects_info, filename="objects_info.csv"):
    # CSV 파일에 저장할 필드명 정의
    fieldnames = ['class_id', 'class_name', 'bbox', 'confidence']
    
    # CSV 파일을 쓰기 모드로 열기
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 헤더 작성
        writer.writeheader()
        
        # objects_info에 있는 각 객체 정보를 CSV 파일에 기록
        for obj in objects_info:
            writer.writerow({
                'class_id': obj['class_id'],
                'class_name': obj['class_name'][0],
                'bbox': str(obj['bbox']),  # 바운딩박스는 문자열로 저장
                'confidence': obj['confidence']
            })

    print(f"Objects information saved to {filename}")


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
              image,
              text,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    data_info = dict(img_id=0, img=cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), texts=texts)
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

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'])

    # 객체 정보 추출
    objects_info = []
    for bbox, label, score in zip(detections.xyxy, detections.class_id, detections.confidence):
        obj_info = {
            'class_id': label.item(),
            'class_name': texts[label.item()][0],
            'bbox': bbox.tolist(),  # 바운딩 박스를 리스트로 변환
            'confidence': score.item()  # confidence score를 추출
        }
        objects_info.append(obj_info)

    return objects_info  # 반환값을 이미지가 아니라 객체 정보 리스트로 변경



def export_model(runner, text, max_num_boxes, score_thr, nms_thr):

    backend = EM.MMYOLOBackend.ONNXRUNTIME
    postprocess_cfg = ConfigDict(pre_top_k=10 * max_num_boxes,
                                 keep_top_k=max_num_boxes,
                                 iou_threshold=nms_thr,
                                 score_threshold=score_thr)

    base_model = runner.model

    texts = [[t.strip() for t in text.split(',')] + [' ']]
    base_model.reparameterize(texts)
    deploy_model = EM.DeployModel(baseModel=base_model,
                                  backend=backend,
                                  postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    device = (next(iter(base_model.parameters()))).device
    fake_input = torch.ones([1, 3, 640, 640], device=device)
    deploy_model(fake_input)

    save_onnx_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('pth', 'onnx'))
    # export onnx
    with BytesIO() as f:
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
        torch.onnx.export(deploy_model,
                          fake_input,
                          f,
                          input_names=['images'],
                          output_names=output_names,
                          opset_version=12)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, save_onnx_path)
    return gr.update(visible=True), save_onnx_path


# Gradio 인터페이스와 게임 로직 통합
import gradio as gr

import pandas as pd
import os


def demo(runner, args):

    question_count = 0  # 질문 횟수 초기화
    previous_candidates = None 

    with gr.Blocks(title="YOLO-World") as demo:
        with gr.Row():  # YOLO-World: Real-Time Open-Vocabulary
            gr.Markdown('<h1><center>Intelligent Robot Project YOLO-World: Real-Time Open-Vocabulary</center></h1>')

        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    image = gr.Image(type='pil', label='Input image')

                input_text = gr.Textbox(
                    lines=7,
                    label='Enter the classes to be detected, separated by comma',
                    value=', '.join(CocoDataset.METAINFO['classes']),
                    elem_id='textbox'
                )

                # 사용자 질문 입력 필드 추가
                question_text = gr.Textbox(
                    lines=2,
                    label="Ask your question (e.g., 'Is it a place to sit down?')",
                    placeholder="Type your question here",
                    elem_id="question-textbox"
                )

                with gr.Row():
                    submit = gr.Button('Submit')
                    clear = gr.Button('Clear')

                with gr.Row():
                    export = gr.Button('Deploy and Export ONNX Model')

                with gr.Row():
                    gr.Markdown("It takes a few seconds to generate the ONNX file!")

                out_download = gr.File(visible=False)

                max_num_boxes = gr.Slider(minimum=1, maximum=300, value=100, step=1, interactive=True, label='Maximum Number Boxes')
                score_thr = gr.Slider(minimum=0, maximum=1, value=0.05, step=0.001, interactive=True, label='Score Threshold')
                nms_thr = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.001, interactive=True, label='NMS Threshold')

            with gr.Column(scale=0.7):
                output_image = gr.Image(type='pil', label='Output image')
                output_text = gr.Textbox(label="Response Text", interactive=False)
                updated_candidates = gr.State()
        
        # 최초 객체 탐지 및 질문 응답 처리 함수 수정
        def process_question(image, question_text, input_text, max_num_boxes, score_thr, nms_thr):
            nonlocal previous_candidates, question_count  # 이전 후보와 질문 횟수 추적

            # 최초 객체 탐지 (첫 번째 질문이 제출되기 전에만 실행)
            if question_count == 0:
                inferred_objects = run_image(runner, image, "'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'",
                max_num_boxes, score_thr, nms_thr)
                # input_text = "'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'"
                
                print('first inferred_objects',inferred_objects)
                question_count += 1  # 첫 번째 객체 탐지 후, 질문 횟수 증가
                
                # 탐지된 객체를 CSV에 기록
                update_object_info_csv(inferred_objects)
                # 초기화 및 객체 탐지
                objects_info = load_objects_info_from_csv("objects_info.csv")
                target_object = objects_info[0]  # 랜덤으로 정답 객체 선정
                previous_candidates = objects_info  # 초기 정답 후보는 전체 탐지된 객체들


            # 질문에서 클래스명을 추출하기
            detected_class_name = re.search(r"Is it (.+)\?", question_text)
            if detected_class_name:
                detected_class_name = detected_class_name.group(1)
            else:
                return image, "Invalid question format. Please ask a valid question.", previous_candidates

            # 모델 탐지 및 후보 설정
            inferred_objects = run_image(runner, image, detected_class_name, max_num_boxes, score_thr, nms_thr)
            visualize_bounding_boxes(image, inferred_objects)  # 여기에 객체 정보를 전달합니다.
            print('inferred_objects',inferred_objects)
            # 후보 갱신
            updated_candidates = intersect_objects(previous_candidates, inferred_objects)
            print('updated_candidates',updated_candidates)
            # 후보 시각화
            visualize_bounding_boxes(image, updated_candidates)

            # 질문 횟수 증가
            question_count += 1

            # 게임 종료 조건: 정답 후보가 하나로 좁혀지면
            if len(updated_candidates) == 1:
                result = f"You win! The answer is: {updated_candidates[0]['class']}"
                return image, result, updated_candidates

            # 10번 질문 후에도 정답 후보가 1개로 좁혀지지 않으면
            if question_count >= 10:
                result = f"You lose! The correct answer was: {target_object['class']}"
                return image, result, updated_candidates

            # 후보 업데이트
            previous_candidates = updated_candidates
            return image, f"Question {question_count}: {detected_class_name}?", updated_candidates

        submit.click(process_question,
                     [image, question_text, input_text, max_num_boxes, score_thr, nms_thr],
                     [output_image, output_text, previous_candidates])
        clear.click(lambda: [None, '', None], None,
                    [image, input_text, output_image])

        export.click(partial(export_model, runner),
                     [input_text, max_num_boxes, score_thr, nms_thr],
                     [out_download, out_download])

        demo.launch(server_name='0.0.0.0',
                    server_port=8080)  # port 80 does not work for me

if __name__ == '__main__':
    args = parse_args()

    # load config
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

    


