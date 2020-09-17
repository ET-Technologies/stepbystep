'''
python3 load_video.py
'''

#from imutils.video import VideoStream
#from imutils.video import FPS
import numpy as np
import argparse
#import imutils
import pickle
import time
import cv2
import os

#video = 'demo.mp4'
video = 'image.png'


def example_videocapture_01():
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print("load frame")
        print (video)
        cv2.imwrite('test.png', frame)
        
        return frame

def example_videocapture_02():
    cap = cv2.VideoCapture(video)
    while True:
        for _ in range(10):
            _, frame=cap.read()
            print("load frame")
            print (video)
            cv2.imwrite('test.png', frame)
        yield frame
            
def example_videocapture_03():
    cap = cv2.VideoCapture(video)
    while True:
        for _ in range(10):
            _, frame=cap.read()
            print("load frame")
            print (video)
            cv2.imwrite('test.png', frame)
        return frame
        
def example_videocapture_04():
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print("load frame")
        print (video)
        cv2.imwrite('test.png', frame)
        
        return frame
            
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
    
def main ():
    #example_videocapture()
    #frame = example_videocapture_02()
    frame = example_videocapture_04()
    print (frame)
    cv2.imwrite('test02.png', frame)

# Start program
if __name__ == '__main__':
    main()