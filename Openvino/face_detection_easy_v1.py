'''
# Udacity Workspace
# source /opt/intel/openvino/bin/setupvars.sh
# Model Downloader python3 downloader.py --name face-detection-retail-0004 --precisions FP32 -o /home/workspace
# python3 face_detection.py --model models/face-detection-retail-0004 --video demo.mp4

Linux
source /opt/intel/openvino/bin/setupvars.sh

python3 ./Openvino/face_detection_easy_v1.py \
--model models/fp16/face-detection-adas-0001 \
--video demos/demo.mp4

# Raspberry
# source /opt/intel/openvino/bin/setupvars.sh
#python3 face_detection_easy_v1.py --model models/face-detection-adas-0001 --video demos/demo.mp4 --device MYRIAD
'''

import numpy as np
import time
import os
import cv2
import argparse
import sys
from os import path
from openvino.inference_engine import IENetwork, IECore
import logging as log

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device, extensions, version):

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions
        self.threshold = threshold
        self.version = version
        print("--------")
        print("model_weights: " + str(self.model_weights))
        print("model_structure: " + str(self.model_structure))
        print("device: " + str(self.device))
        print("extensions: " + str(self.extensions))
        print("--------")



    def load_model(self, device, extension):

        # Initialise the network and save it in the self.model variables
        try:
            log.info("Reading model ...")
            self.model = IENetwork(model=self.model_structure, weights=self.model_weights)
            #self.model = IECore.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError ("Could not initialise the network")

        print("--------")
        print("Model is loaded as self.model: " + str(self.model))
        
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.exec_network = None

        print("input_name: " + str(self.input_name))
        print("input_shape: " + str(self.input_shape))
        print("output_name: " + str(self.output_name))
        print("output_shape: " + str(self.output_shape))
        print("--------")
        self.core = IECore()

        # Add extension
        #CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
        #if "CPU" in self.device and self.version == 2019
        #    log.info("Add extension: ({})".format(str(CPU_EXTENSION)))
        #    self.core.add_extension(CPU_EXTENSION, device)
        
        self.exec_network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print("Exec_network is loaded as:" + str(self.exec_network))
        
        ### TODO: Check for supported layers ###
    #    if "CPU" in device:
     #       supported_layers = self.core.query_network(self.exec_network, "CPU")
      #      print("supported_layers: " + str(supported_layers)) 
       #     not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        #    print("not_supported_layers: " + str(not_supported_layers)) 
         #   if len(not_supported_layers) != 0:
          #      sys.exit(1)
        
        print("--------")

    def predict(self, frame, initial_w, initial_h):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        print("--")
        print("Start predictions")
        self.width = initial_w
        self.height = initial_h
        requestid = 0
        preprocessed_image = self.preprocess_input(frame)
        # Starts synchronous inference
        print("Start syncro inference")
        log.info("Start syncro inference")
        outputs = self.exec_network.infer({self.input_name: preprocessed_image})
        print("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        print("Output of the inference request (self.output_name): " + str(outputs))
        processed_image = self.boundingbox(outputs, frame)
        print("End predictions")
        print("--------")
        cv2.imshow("Test", processed_image)

        return processed_image

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, frame):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print("--------")
        print("Start: preprocess image")
        log.info("Start: preprocess image")
        n, c, h, w = (self.core, self.input_shape)[1]
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        print("Original image size is W= ({}) x H= ({})".format(str(self.width),str(self.height)))
        print("Image is now [BxCxHxW]: " + str(image.shape))
        print("End: preprocess image")
        print("--------")
        
        return image

    def boundingbox(self, outputs, frame):
        #coords = []
        print("--------")
        print("Start: boundingbox")
        print("Bounding box input: " + str(outputs))
        #print("Coords: " + str(coords))
        print("Original image size is (W x H): " + str(self.width) + "x" + str(self.height))
        for obj in outputs[0][0]:
            if obj[2] > self.threshold:
                obj[3] = int(obj[3] * self.width)
                obj[4] = int(obj[4] * self.height)
                obj[5] = int(obj[5] * self.width)
                obj[6] = int(obj[6] * self.height)
                #coords.append([obj[3], obj[4], obj[5], obj[6]])
                cv2.rectangle(frame, (obj[3], obj[4]), (obj[5], obj[6]), (0, 55, 255), 1)
                print("Bounding box output coordinates of frame: " + str(obj[3]) + " x " + str(obj[4]) + " x " + str(obj[5]) + " x " + str(obj[6]))
                self.xmin = int(obj[3])
                self.ymin = int(obj[4])
                self.xmax = int(obj[5])
                self.ymax = int(obj[6])
                

        print("End: boundingbox")
        print("--------")
        frame_cropped = frame.copy()
        #frame_cropped = frame_cropped[self.ymin:(self.ymax + 1), self.xmin:(self.xmax + 1)]
        cv2.imwrite("cropped image.png", frame_cropped)
        cv2.imshow("Test", frame_cropped)
        self.preprocess_output(frame)

        #frame = self.cropimage(frame)

        return frame

    def preprocess_output(self, frame):
        # crop image to fit the next model
        print("--------")
        print("Start: preprocess_output")
        print("Coordinates for cropped frame are xmin x ymin x xmax x ymax: " + str(
            self.xmin) + " x " + str(self.ymin) + " x " + str(self.xmax) + " x " + str(self.ymax))
        frame_cropped = None
        frame_cropped = frame[self.ymin:(self.ymax + 1), self.xmin:(self.xmax + 1)]
        cv2.imwrite("cropped_image.png", frame_cropped)
        print("--------")
        print("End: preprocess_output")
        return

    def videofile(self, video):
        if video =='video':
            input_type = 'video'
        else:
            input_type ='cam'

        return input_type

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extension', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--version', default=2020)

    return parser


def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    extension = args.extension
    video = args.video
    output_path = args.output_path
    threshold = args.threshold
    version = args.version


    # Load class Model_X
    inference = Model_X(model_name, threshold, device, extension, version)
    print("Load class Model_X = OK")
    print("--------")

    # Loads the model
    inference.load_model(device,extension)
    print("Load Model = OK")
    print("--------")
    cap = cv2.VideoCapture(video)

    # Get the input video stream
    try:
        print("Reading video file", video)
        cap = cv2.VideoCapture(video)
        cap.open(video)
        if not path.exists(video):
            print("Cannot find video file: " + video)
    except FileNotFoundError:
        print("Cannot find video file: " + video)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    # Capture information about the input video stream
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("--------")
    print("Input video Data")
    print("initial_w: " + str(initial_w))
    print("initial_h: " + str(initial_h))
    print("video_len: " + str(video_len))
    print("fps: " + str(fps))
    print("--------")

    # Define output video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('output_video3.mp4', fourcc, fps, (initial_w, initial_h))

    try:
        while cap.isOpened():
            result, frame = cap.read()
            if not result:
                break
            image = inference.predict(frame, initial_w, initial_h)
            print("The video is writen to the output path")
            out_video.write(image)
    except Exception as e:
        print("Could not run Inference: ", e)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    log.basicConfig(filename="logging.txt", level=log.INFO)
    log.info("Start logging")
    main()
