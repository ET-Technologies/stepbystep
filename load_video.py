from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

video = 'demos\demo.mp4'
video = 'demos\image.jpg'


def example_videocapture():
    cap = cv2.VideoCapture(args["video"])
    while True:
        ret, frame = cap.read()

        if not cap:
            break

        frame = imutils.resize(frame, width=450)
        (h, w) = frame.shape[:2]
        print (h,w)

        cv2.imshow("Test", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def example_videostream():
    cap = VideoStream(usePiCamera=True).start()
    cap = VideoStream(0).start()
    # camera warm up
    time.sleep(2.0)
    while True:
        frame = cap.read()
        frame = imutils.resize(frame, width=450)
        (h, w) = frame.shape[:2]
        print (h,w)

        cv2.imshow("Test", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.stop()





