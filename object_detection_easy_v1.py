#python object_detection.py --model person-detection-retail-0013 --video Manufacturing.mp4

import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore

def build_argparser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='results/')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    return parser
