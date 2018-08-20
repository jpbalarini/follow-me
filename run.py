# -*- coding: utf-8 -*-
import sys
import argparse
from lib.yolo.yolo import YOLO, detect_video
from PIL import Image
import glob
import cv2
import numpy as np

FLAGS = None

def detect_img(yolo, image_path):
    try:
        image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        opencvImage = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", opencvImage)


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

    parser.add_argument(
        '--images_path', type=str,
        help='Path from where to read images', default='/Volumes/BACKUP_2TB/Maestria/Datasets/Crowd_PETS09/S2/L1/Time_12-34/View_001'
    )

    FLAGS = parser.parse_args()

    print('Going to run first stage of the pipeline...')

    yolo = YOLO(**vars(FLAGS))
    for image_path in glob.glob(FLAGS.images_path + '/*.jpg'):
        detect_img(yolo, image_path)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

