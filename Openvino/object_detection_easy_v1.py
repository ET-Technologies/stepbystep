#python object_detection.py --model person-detection-retail-0013 --video Manufacturing.mp4

import numpy as np
import time
import os
import cv2
import argparse
import sys
from openvino.inference_engine import IENetwork, IECore

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

# Start asyncron inference request and wait for the result
def async_inference(exec_network, input_name, image):
    infer_request_handle = exec_network.start_async(request_id=0, inputs={input_name: image})
    
    while True:
        status = exec_network.requests[0].wait(-1)
        if status == 0:
            break
        else:
            time.sleep(1)
        print ("status: " +str(status))
        
        return infer_request_handle

# Wait for the result (is implemented in "Start asyncron inference request")
def wait(exec_network, request_id=0):
    wait_process = exec_network.requests[request_id].wait(1)
    
    return wait_process

# Get the inference output
def get_output(exec_network, infer_request_handle, output_name, request_id=0, output=None):
    if output:
        res = infer_request_handle.output[output]
    else:
        res = exec_network.requests[request_id].outputs[output_name]
    return res
        
# Draw Bounding Box
def boundingbox(res, initial_w, initial_h, frame, threshold):
    current_count = 0
    for obj in res[0][0]:
    # Draw bounding box for object when it's probability is more than the specified threshold
        if obj[2] > threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
            print ("Current count: " + str(current_count))
    return frame            

def main():
    args = build_argparser().parse_args()
    model_name=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    
    ## Load Model
    model_weights=model_name+'.bin'
    model_structure=model_name+'.xml'
    # Loading time
    start_model_load_time = time.time()
        # Extension
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    
    # Read the model
    core = IECore()
    #model = core.read_network(model=model_structure, weights=model_weights) # new openvino version
    model = IENetwork(model=model_structure, weights=model_weights) #old openvino version
        # Add CPU extension
    core.add_extension(CPU_EXTENSION, device)
    
    # Load the network into an executable network
    exec_network = core.load_network(network=model, device_name=device, num_requests=1)
    print ("Model is loaded")
    
    # Time to load the model
    total_model_load_time = time.time() - start_model_load_time
    print ("Time to load model: " + str(total_model_load_time))
    
    # Get the input layer
    input_name = next(iter(model.inputs))
    input_shape = model.inputs[input_name].shape
    output_name = next(iter(model.outputs))
    output_shape = model.outputs[output_name].shape

    print ("input_name:" +str(input_name))
    print ("input_shape:" +str(input_shape))
    print ("output_name: " +str(output_name))
    print ("output_shape: " +str(output_shape))
    
    # Get the input shape
    n, c, h, w = (core, input_shape)[1]
    
    # Get the input video stream
    cap=cv2.VideoCapture(video_file)
    
    # Information about the input video stream
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
    request_id=0
    
    ### Read from the video capture ###
    while cap.isOpened():
        ret, frame=cap.read()
        if not ret:
            break
        key_pressed = cv2.waitKey(60)
        
        # Pre-process the image as needed
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        print ("n-c-h-w " + str(n) + "-" + str(c) + "-" +str(h) + "-" +str(w))
        
        # Start asynchronous inference for specified request
        start_inference_time = time.time()
        infer_request_handle = async_inference(exec_network, input_name, image)
        
        # Get the output data
        res = get_output(exec_network, infer_request_handle, output_name, request_id=0, output=None)
        detection_time = time.time() - start_inference_time
        print ("Detection time: " + str(detection_time))
        
        # Draw Bounding Box
        frame = boundingbox(res, initial_w, initial_h, frame, threshold)
        
        # Write the output video
        out_video.write(frame)
        
    cap.release()
    cv2.destroyAllWindows()
    

# Start sequence
if __name__=='__main__':
    main()
