
from pywget import wget
Linux
source /opt/intel/openvino/bin/setupvars.sh

# Linux
cd /opt/intel/openvino/deployment_tools/tools/model_downloader
Model Downloader 
python3 downloader.py --name face-detection-retail-0004 --precisions FP32 -o /home/workspace
python3 downloader.py --name gaze-estimation-adas-0002 -o /home/thomas/Models
# python3 face_detection.py --model models/face-detection-retail-0004 --video demo.mp4