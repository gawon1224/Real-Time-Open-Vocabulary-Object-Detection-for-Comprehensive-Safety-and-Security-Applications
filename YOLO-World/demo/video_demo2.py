import argparse
import cv2
import torch
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmengine.utils import track_iter_progress
from mmyolo.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World video demo')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('video', help='video file path')
    parser.add_argument(
        'text',
        help=
        'text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference')
    parser.add_argument('--score-thr',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--out', type=str, help='output video file')
    args = parser.parse_args()
    return args


def inference_detector(model, image, texts, test_pipeline, score_thr=0.3):
    data_info = dict(img_id=0, img=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]
    output.pred_instances = pred_instances
    return output


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    # Reparameterize texts
    model.reparameterize(texts)

    # Init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # Open video using cv2.VideoCapture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {args.video}")

    # Prepare for video writing if output is specified
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames

        result = inference_detector(model, frame, texts, test_pipeline, score_thr=args.score_thr)
        visualizer.add_datasample(name=f'frame_{frame_idx}',
                                  image=frame,
                                  data_sample=result,
                                  draw_gt=False,
                                  show=False,
                                  pred_score_thr=args.score_thr)
        frame = visualizer.get_image()

        if args.out:
            video_writer.write(frame)

        frame_idx += 1

    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()


if __name__ == '__main__':
    main()
