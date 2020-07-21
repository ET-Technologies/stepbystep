#Downloader

cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

source /opt/intel/openvino/bin/setupvars.sh


## Udacity Workspace
sudo ./downloader.py --name human-pose-estimation-0001 -o /home/workspace 
## Mac
sudo ./downloader.py --name resnet50-binary-0001 -o /Users/pro/PycharmProjects/Udacity/models

### Options
--precisions FP32 


### Models on Openvino

#### Classification
resnet50-binary-0001
resnet18-xnor-binary-onnx-0001

#### Object Detection
face-detection-retail-0004
face-detection-0105
product-detection-0001
