# python text_detection_easy_v1.py --model intel/text-detection-0003/FP32/text-detection-0003 --video images/sign.jpg --output_path outputs
#intel/text-recognition-0012/FP16/text-recognition-0012.xml
#images/sitting-on-car.jpg
#images/car.png
#text-recognition-0012
#intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml
#demo.mp4
#images/sign.jpg
#text-detection-0003
#python downloader.py --name text-detection-0003 -o /home/workspace
#intel/text-detection-0003/FP32/text-detection-0003.xml

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
    def predict(self, image, initial_w, initial_h):
        request_id=0

        # save original image
        input_img = image
        # Pre-process the image
        image = self.preprocess_input(image)
        print("--------")
        print ("Start syncro inference")
        result = self.exec_network.infer({self.input_name:image}) #syncro inference
        print("result syncro inference: " + str(result))
        result_request = self.exec_network.requests[request_id].outputs[self.output_name]
        print("result_request: " + str(result_request))
        processed_result = self.handle_text(result_request, image.shape)
        
        #processed_result = self.get_output(processed_result, 0, output=None)
        print("result: get output" + str(result))
        #text detection
        image = self.text_detection(input_img, result)
        
        return image

    # Preprocess the image
    def preprocess_input(self, image):
        # Get the input shape
        print("--------")
        print ("Start preprocess input")
        n, c, h, w = (self.core, self.input_shape)[1]
        print("n-c-h-w " + str(n) + "-" + str(c) + "-" + str(h) + "-" + str(w))
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        print("Image is now: " + str(n) +("-")+str(c)+("-")+str(h)+("-")+str(w))
        print ("End of preprocess input")
        print("--------")
        
        return image

    # Get the inference output
    def get_output(self, infer_request_handle, request_id, output):
        if output:
            res = infer_request_handle.output[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_name]
        return res
    
    def get_mask(processed_output):
        '''
        Given an input image size and processed output for a semantic mask,
        returns a masks able to be combined with the original image.
        '''
    
        # Create an empty array for other color channels of mask
        empty = np.zeros(processed_output.shape)
        # Stack to make a Green mask where text detected
        mask = np.dstack((empty, processed_output, empty))

        return mask
    
    def text_detection(self, input_img, result):
        
        # Get only text detections above 0.5 confidence, set to 255
        print("--------")
        print ("Start text detection")
        output = np.where(result[1]>0.5, 255, 0)
        print("output: " + str(output))
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = input_img + text_mask
        print ("End text detection")
        return image
    
    def handle_text(self, result, input_shape):
        '''
        Handles the output of the Text Detection model.
        Returns ONLY the text/no text classification of each pixel,
            and not the linkage between pixels and their neighbors.
        '''
        # TODO 1: Extract only the first blob output (text/no text classification)
        print("--------")
        print ("Start handle text")
        print("Input: result keys: handle text()" + str(result.keys()))
        text_classes = result['model/segm_logits/add']
        print("text_classes: " + str(text_classes))
    
        # TODO 2: Resize this output back to the size of the input
        out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
        for t in range(len(text_classes[0])):
                out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])
        print("out_text: handle text()" + str(out_text))
        print ("End handle_text")
        print("--------")

        return out_text
        
    
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
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

    start_model_load_time = time.time()  # Time to load the model (Start)
    # Load class Inference as inference
    inference = Inference(model_name, device, CPU_EXTENSION)
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
            image = inference.predict(frame, initial_w, initial_h)
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Read the input image
    image = cv2.imread(video_file)
    # Scale the output text by the image shape
    scaler = max(int(image.shape[0] / 1000), 1)
    # Write the text of color and type onto the image
    image = cv2.putText(image, "Color: {}, Type: {}".format(color, car_type), (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 2 * scaler, (255, 255, 255), 3 * scaler)

# Start sequence
if __name__ == '__main__':
    main()
