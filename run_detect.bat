@echo off
cd yolov5-6.0
python detect.py --weights best.pt --source ../datasets/wangy/images/train --img 640 --conf 0.25
pause 