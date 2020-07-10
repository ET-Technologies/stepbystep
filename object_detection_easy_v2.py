#python object_detection_v2.py --model person-detection-retail-0013 --video Manufacturing.mp4
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
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        
        # Initialise the network and save it in the self.model variables
        self.model=IENetwork(self.model_structure, self.model_weights) # old openvino version
        #self.model = core.read_network(self.model_structure, self.model_weights) # new openvino version
        
        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

        print ("input_name:" +str(self.input_name))
        print ("input_shape:" +str(self.input_shape))
        print ("output_name: " +str(self.output_name))
        print ("output_shape: " +str(self.output_shape))

    # Loads the model
    def load_model(self):
        # Adds Extension
        CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
        self.core = IECore()
        self.core.add_extension(CPU_EXTENSION, self.device)
        # Load the network into an executable network
        self.exec_network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print ("Model is loaded")
        
    # Start inference and prediction    
    def predict(self, image, initial_w, initial_h):
        
        # save original image
        input_img = image
        # Pre-process the image
        image = preprocess_input(image)
   
    def preprocess_input(self, image):
        # Get the input shape
        n, c, h, w = (self.core, self.input_shape)[1]
        print ("n-c-h-w " + str(n) + "-" + str(c) + "-" +str(h) + "-" +str(w))
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
    return image
        
        
    # Start asyncron inference request and wait for the result
    def async_inference(self, image):
        infer_request_handle = self.exec_network.start_async(request_id=0, inputs={self.input_name: image})
        while True:
            status = self.exec_network.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
            print ("status: " +str(status))
            return infer_request_handle
        
     # Get the inference output
    def get_output(infer_request_handle, request_id=0, output=None):
        if output:
            res = infer_request_handle.output[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_name]
        return res
    
    # Draw Bounding Box
    def boundingbox(res, initial_w, initial_h, frame):
        current_count = 0
        for obj in res[0][0]:
            # Draw bounding box for object when it's probability is more than the specified threshold
            if obj[2] > self.threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
                current_count = current_count + 1
                print ("Current count: " + str(current_count))
        return frame 

# Collect all the necessary input values
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

def main():
    args = build_argparser().parse_args()
    model_name=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    
    start_model_load_time=time.time() # Time to load the model (Start)
    # Load class Inference as inference
    inference= Inference(model_name, device, threshold)
    # Loads the model from the inference class
    inference.load_model()
    total_model_load_time = time.time() - start_model_load_time # Time model needed to load
    print ("Time to load model: " + str(total_model_load_time))
    
    # Get the input video stream
    cap=cv2.VideoCapture(video_file)
    
    # Capture information about the input video stream
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print ("initial_w: " +str(initial_w))
    print ("initial_h: " +str(initial_h))
    print ("video_len: " +str(video_len))
    print ("fps: " +str(fps))
    
    # Define output video
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video3.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    # We have just one request number is 0
    request_id=0
    
    ### Read from the video capture ###
    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            
            # Get the image from the inference class
            image= inference.predict(frame, initial_w, initial_h)
            # Write the output video
            out_video.write(image)
    
    cap.release()
    cv2.destroyAllWindows()     
    
    

# Start sequence
if __name__=='__main__':
    main()
