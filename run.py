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

# global variables if I want to read detections from file (MOT format)
detections_in = None
frame_indices = None


def yolo_setup(FLAGS):
    yolo = YOLO(**vars(FLAGS))
    return yolo

def deep_sort_setup():
    encoder = generate_detections.create_box_encoder(deep_sort_model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker

def run_yolo(frame, yolo):
    pil_image = Image.fromarray(frame)
    yolo_image, boxes, scores = yolo.detect_image(pil_image)
    return yolo_image, boxes, scores

def run_deep_sort(frame, detection_boxes, detection_scores, tracker, encoder):
    # deep_sort uses a BGR image as input
    # check generate_detections function description
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    features = encoder(frame, detection_boxes)
    detections = [
        Detection(bbox, score, feature) for bbox, score, feature in zip(detection_boxes, detection_scores, features)
    ]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Update tracker.
    tracker.predict()
    tracker.update(detections)

    return tracker, detections

def run_pipeline(frame, yolo, tracker, encoder, frame_idx=None, run_detection=True):
    if run_detection:
        yolo_image, boxes, scores = run_yolo(frame, yolo)
    else:
        mask = frame_indices == frame_idx
        rows = detections_in[mask]
        boxes = rows[:, 2:6].copy()
        scores = rows[:, 6:7].copy()

    tracker, detections = run_deep_sort(frame, boxes, scores, tracker, encoder)

    return frame, tracker, detections

def display_results(frame, fps, tracker, detections):
    img_height, img_width = frame.shape[:2]

    for track in tracker.tracks:
        if track.is_confirmed() and track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),
            color=(255, 255, 255), thickness=2)
        cv2.putText(frame, text=str(track.track_id), org=(int(bbox[0]), int(bbox[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)

    for det in detections:
        bbox = det.to_tlbr()
        cv2.rectangle(frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),
            color=(255, 0, 0), thickness=2)
        label = "{0:.1f}".format(det.confidence)
        cv2.putText(frame, text=label, org=(int(bbox[2]), int(bbox[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)

    max_window_size = 800, 600
    scale_width = max_window_size[0] / img_width
    scale_height = max_window_size[1] / img_height
    scale = min(scale_width, scale_height)
    window_width = int(img_width * scale)
    window_height = int(img_height * scale)

    cv2.putText(frame, text=fps, org=(5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=(0, 0, 255), thickness=3)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    resized_image = cv2.resize(frame, (window_width, window_height))

    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('result', resized_image)

def compute_fps(accum_time, curr_fps, prev_time):
    curr_time = timer()
    exec_time = curr_time - prev_time
    new_prev_time = curr_time
    new_accum_time = accum_time + exec_time
    new_curr_fps = curr_fps + 1
    if new_accum_time > 1:
        new_accum_time = new_accum_time - 1
        new_fps = "FPS: " + str(new_curr_fps)
        new_curr_fps = 0
    return new_accum_time, new_curr_fps, new_fps, new_prev_time

def main(FLAGS):
    output_path = FLAGS.video_output
    # initial setup
    yolo = yolo_setup(FLAGS)
    encoder, tracker = deep_sort_setup()

    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    run_detection = True

    if FLAGS.load_detection_file:
        run_detection = False
        global detections_in
        detections_in = np.loadtxt(FLAGS.load_detection_file, delimiter=',')
        # Filter those which are below confidence
        detections_in = detections_in[detections_in[:, 6] > FLAGS.score]
        global frame_indices
        frame_indices = detections_in[:, 0].astype(np.int)

    if FLAGS.images_path:
        filenames = glob.glob(FLAGS.images_path + '/*.jpg')
        filenames.sort()
        images = [cv2.imread(image_path) for image_path in filenames]

        for frame_idx, image in enumerate(images, start=1):
            print('Frame index: ' + str(frame_idx))
            # fix image color order BGR (opencv default) -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result_image, tracker, detections = run_pipeline(image, yolo, tracker, encoder, frame_idx, run_detection)
            accum_time, curr_fps, fps, prev_time = compute_fps(accum_time, curr_fps, prev_time)
            display_results(result_image, fps, tracker, detections)

            # show one frame at a time
            cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif FLAGS.video_input:
        vid = cv2.VideoCapture(FLAGS.video_input)
        if not vid.isOpened():
            raise IOError("Can't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

        while True:
            return_value, image = vid.read()
            # fix image color order BGR (opencv default) -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result_image, tracker, detections = run_pipeline(image, yolo, tracker, encoder)
            accum_time, curr_fps, fps, prev_time = compute_fps(accum_time, curr_fps, prev_time)
            display_results(result_image, fps, tracker, detections)

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

    parser.add_argument(
        '--load_detection_file', type=str,
        help='Does not perform classification and loads detections from file (MOT format)', default=""
    )

    parser.add_argument(
        '--score', type=float,
        help='Detections with lower scores than this will be filtered', default=0.3
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
