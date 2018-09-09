# -*- coding: utf-8 -*-
import sys
import argparse
from lib.yolo.yolo import YOLO, detect_video
from PIL import Image
import glob
import cv2
import numpy as np
import os

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

def detect_img(yolo, image_path):
    try:
        image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
    else:
        r_image, boxes = yolo.detect_image(image)
        # opencvImage = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        # cv2.imshow('result', opencvImage)

        # cv2.imwrite(os.path.splitext(image_path)[0] + '_output.jpg', opencvImage)
        return r_image, boxes

def deep_sort(frame, detection_boxes, tracker, encoder):
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

    opencvImage = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', opencvImage)


def main(FLAGS):
    yolo = YOLO(**vars(FLAGS))

    encoder = generate_detections.create_box_encoder(deep_sort_model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    for image_path in glob.glob(FLAGS.images_path + '/*.jpg'):
        image, boxes = detect_img(yolo, image_path)
        deep_sort(np.array(image), boxes, tracker, encoder)

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
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    # /Volumes/BACKUP_2TB/Maestria/Code/a
    parser.add_argument(
        '--images_path', type=str,
        help='Path from where to read images', default='/Volumes/BACKUP_2TB/Maestria/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_001'
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
