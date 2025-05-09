# NB

## FineTuning the YOLOv8: training on the bvn dataset

yolo train model=yolov8n.pt data=yolo-bvn.yaml epochs=50 imgsz=320 batch=8

yolo predict model=./runs/detect/train/weights/best.pt source=30.jpg save=True

## Run with args

python3 scripts/yolov8-stereo.py --input data/dual_camera_20250505_163126.avi

    ```shell
    
        python3 scripts/yolov8-stereo.py --input "image/video path"

    ```
    