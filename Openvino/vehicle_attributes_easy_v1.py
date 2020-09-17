# python vehicle_attributes_easy_v1.py --model models/vehicle-attributes-recognition-barrier-0039 --video truck.jpg --output_path outputs
#truck.jpg
#images/blue-car.jpg
#models/vehicle-attributes-recognition-barrier-0039.xml
# python vehicle_attributes_easy_v1.py --model intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 --video demo.mp4 --output_path outputs
# python vehicle_attributes_easy_v1.py --model models/semantic-segmentation-adas-0001 --video demo.mp4 --output_path outputs
#semantic-segmentation-adas-0001
# python vehicle_attributes_easy_v1.py --model models/human-pose-estimation-0001 --video demo.mp4 --output_path outputs
#python vehicle_attributes_easy_v1.py --model intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 --video images/car.png --output_path outputs

import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore

class Inference:
    '''
    Class with all relevant tools to do object detection
    '''

    # Load all relevant variables into the class
    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold

        # Initialise the network and save it in the self.model variables
        self.model = IENetwork(self.model_structure, self.model_weights)  # old openvino version
        # self.model = core.read_network(self.model_structure, self.model_weights) # new openvino version

        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        self.input_name_all = [i for i in self.model.inputs.keys()] # gets all input_names
        self.input_name_all_02 = self.model.inputs.keys() # gets all output_names
        self.input_name_first_entry = self.input_name_all[0]
        
        self.input_shape = self.model.inputs[self.input_name].shape
        
        self.output_name = next(iter(self.model.outputs))
        self.output_name_type = self.model.outputs[self.output_name]
        self.output_names = [i for i in self.model.outputs.keys()] # gets all output_names
        self.output_names_total_entries = len(self.output_names)
        
        self.output_shape = self.model.outputs[self.output_name].shape
        self.output_shape_second_entry = self.model.outputs[self.output_name].shape[1]
        self.output_name_first_entry = self.output_names[0]
        
        print("--------")
        print("input_name: " + str(self.input_name))
        print("input_name_all: " + str(self.input_name_all))
        print("input_name_all_total: " + str(self.input_name_all_02))
        print("input_name_first_entry: " + str(self.input_name_first_entry))
        print("--------")
        
        print("input_shape: " + str(self.input_shape))
        print("--------")
        
        print("output_name: " + str(self.output_name))
        print("output_name type: " + str(self.output_name_type))
        print("output_names: " +str(self.output_names))
        print("output_names_total_entries: " +str(self.output_names_total_entries))
        print("output_name_first_entry: " + str(self.output_name_first_entry))
        print("--------")
        
        print("output_shape: " + str(self.output_shape))
        print("output_shape_second_entry: " + str(self.output_shape_second_entry))
        print("--------")
        

    # Loads the model
    def load_model(self):
        # Adds Extension
        CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
        self.core = IECore()
        self.core.add_extension(CPU_EXTENSION, self.device)
        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print("Model is loaded")

    # Start inference and prediction
    def predict(self, image):

        # save original image
        input_img = image
        # Pre-process the image
        image = self.preprocess_input(image)
        result = self.exec_network.infer({self.input_name:image}) #syncro inference
        print ("Start syncro inference")
        #infer_request_handle = self.async_inference(image)
        #res = self.get_output(infer_request_handle, 0, output=None)
        
        # Vehicle output
        color, car_type = self.vehicle_attributes(result)

        return frame, color, car_type

    # Preprocess the image
    def preprocess_input(self, frame):
        # Get the input shape
        n, c, h, w = (self.core, self.input_shape)[1]
        print("n-c-h-w " + str(n) + "-" + str(c) + "-" + str(h) + "-" + str(w))
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        print ("End of preprocess input")
        
        return image

    # Get the inference output
    def get_output(self, infer_request_handle, request_id, output):
        if output:
            res = infer_request_handle.output[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_name]
        return res

    def vehicle_attributes(self,result):
        #Gets the output of the vehicle model
        
        CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
        
        color = result['color']
        color_flatten = result['color'].flatten()
        color_total = len(color_flatten)
        color_class = np.argmax(color)
        color_class_text = CAR_COLORS[color_class]
        print("--------")
        print("color: " + str(color))
        print("color_flatten: " + str(color_flatten))
        print("total number of colors: " + str(color_total))
        print("color number with the higest propability (argmax): " + str(color_class))
        print("color text with the higest propability (argmax): " + str(color_class_text))
        print("--------")

        CAR_TYPES = ["car", "bus", "truck", "van"]
        
        car_type =result['type']
        car_type_flatten = result['type'].flatten()
        car_type_total = len(car_type_flatten)
        car_type_class = np.argmax(car_type)
        car_type_class_text = CAR_TYPES[car_type_class]
        print("car_type: " + str(car_type))
        print("car_type_flatten: " + str(car_type_flatten))
        print("total number of car types: " + str(car_type_total))
        print("car type with the higest propability (argmax): " + str(car_type_class))
        print("car text with the higest propability (argmax): " + str(car_type_class_text))
        print("--------")

        return color_class, car_type_class
    
# Collect all the necessary input values
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='results/')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    return parser


def main():
    args = build_argparser().parse_args()
    model_name = args.model
    device = args.device
    video = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path
    color = None
    car_type = None

    start_model_load_time = time.time()  # Time to load the model (Start)
    # Load class Inference as inference
    inference = Inference(model_name, device, threshold)
    # Loads the model from the inference class
    inference.load_model()
    total_model_load_time = time.time() - start_model_load_time  # Time model needed to load
    print("Time to load model: " + str(total_model_load_time))
    
    # Get the input video stream
    cap = cv2.VideoCapture(video)
    # Capture information about the input video stream
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print ("initial_w: " +str(initial_w))
    print ("initial_h: " +str(initial_h))
    print ("video_len: " +str(video_len))
    print ("fps: " +str(fps))
    
    while cap.isOpened():
        result, frame = cap.read()
        image = inference.predict(frame)
    #image, color, car_type = inference.predict(frame)
    
    # Read the input image
    image = cv2.imread(video_file)
    # Scale the output text by the image shape
    scaler = max(int(image.shape[0] / 1000), 1)
    # Write the text of color and type onto the image
    image = cv2.putText(image, "Color: {}, Type: {}".format(color, car_type), (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 2 * scaler, (255, 255, 255), 3 * scaler)

# Start sequence
if __name__ == '__main__':
    main()
