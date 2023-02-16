from roboflow import Roboflow
import yolov8
rf = Roboflow(api_key="1ud92IsBdk8IisByhxxw")
project = rf.workspace("eagle-eye").project("basketball-1zhpe")
dataset = project.version(1).download("yolov8")

yolo task=detect mode=train model=yolov8n.pt data="{path to data.yaml}" epochs=100 device=0