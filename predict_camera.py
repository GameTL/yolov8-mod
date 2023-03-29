from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils.torch_utils import select_device
import torch
import yaml
import json
[X_RESOLUTION, Y_RESOLUTION, VIDEO_FPS] = [1280, 720, 30]

model = YOLO("yolov8n-seg.pt")
# model = YOLO("runs/detect/train/weights/best.pt")

with open("ultralytics/yolo/data/datasets/coco8-seg.yaml", "r") as stream:
    try:
        datasets = yaml.safe_load(stream)
        datasets_names = datasets['names']
    except:
        print("No file found")
        datasets_names = ""

start = time.time()

# Camera Config
cap = cv2.VideoCapture(0)
cap.set(3, X_RESOLUTION)
cap.set(4, Y_RESOLUTION)
cap.set(5, VIDEO_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)),
                (10, int(cap.get(4)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    # print("fps: " + str(round(1 / (time.time() - start), 2)))
    start = time.time()

    #TODO Work on this
    results = model.predict(source=frame, conf=0.5, show=True)[0]
    if results.boxes:
        # print(f"DETECT {len(results.boxes)}")
        output = dict()
        for i, obj in enumerate(results.boxes):
            x, y, w, h = obj.xywhn.cpu().numpy()[0]
            name = datasets_names[int(
                obj.cls.cpu().numpy())] if datasets_names else 'unknown'
            output[i] = [name, x, y, w, h]
    # print(json.dumps(output, indent=4))
        print(output)
    #TODO Work on this

    # cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
""" 
{0: ['person', 0.4546875, 0.6409722, 0.51875, 0.69861114]}
 """