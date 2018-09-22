# -*- coding: utf-8 -*-
import sys
import argparse
from lib.yolo.yolo import YOLO, detect_video
from PIL import Image
import glob
import cv2
import numpy as np
import os
from timeit import default_timer as timer

from lib.deep_sort.application_util import preprocessing
from lib.deep_sort.deep_sort import nn_matching
from lib.deep_sort.deep_sort.detection import Detection
from lib.deep_sort.deep_sort.tracker import Tracker
from lib.deep_sort.deep_sort.detection import Detection
from lib.deep_sort.tools import generate_detections

FLAGS = None
# deep sort config
deep_sort_model_filename = 'lib/deep_sort/model_data/mars-small128.pb'
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0


def yolo_setup(FLAGS):
    yolo = YOLO(**vars(FLAGS))
    return yolo

def deep_sort_setup():
    encoder = generate_detections.create_box_encoder(deep_sort_model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker

def run_yolo(yolo, image):
    image, boxes = yolo.detect_image(image)
    return image, boxes

def run_deep_sort(image, detection_boxes, tracker, encoder, fps=None):
    img_width, img_height = image.size
    frame = np.asarray(image)

    features = encoder(frame, detection_boxes)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(detection_boxes, features)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Update tracker.
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if track.is_confirmed() and track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

    for det in detections:
        bbox = det.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

    max_window_size = 800, 600
    scale_width = max_window_size[0] / img_width
    scale_height = max_window_size[1] / img_height
    scale = min(scale_width, scale_height)
    window_width = int(img_width * scale)
    window_height = int(img_height * scale)

    if fps:
        cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2.0, color=(255, 0, 0), thickness=2)
    else:
        # image doesnt have right color order
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    resized_image = cv2.resize(frame, (window_width, window_height))
    cv2.imshow('result', resized_image)

def run_pipeline(image, yolo, tracker, encoder, fps=None):
    new_image, boxes = run_yolo(yolo, image)
    run_deep_sort(image, boxes, tracker, encoder, fps)

def main(FLAGS):
    output_path = FLAGS.video_output
    # initial setup
    yolo = yolo_setup(FLAGS)
    encoder, tracker = deep_sort_setup()

    if FLAGS.images_path:
        for image_path in glob.glob(FLAGS.images_path + '/*.jpg'):
            run_pipeline(image, yolo, tracker, encoder)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif FLAGS.video_input:
        vid = cv2.VideoCapture(FLAGS.video_input)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            run_pipeline(image, yolo, tracker, encoder, fps)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    yolo.close_session()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    # /Volumes/BACKUP_2TB/Maestria/Datasets/test.mp4
    parser.add_argument(
        "--video_input", nargs='?', type=str, required=False,
        default='/Volumes/BACKUP_2TB/Maestria/Datasets/test.mp4',
        help = "Video input path"
    )
    parser.add_argument(
        "--video_output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    # /Volumes/BACKUP_2TB/Maestria/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_001
    parser.add_argument(
        '--images_path', type=str,
        help='Path from where to read images', default=""
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
