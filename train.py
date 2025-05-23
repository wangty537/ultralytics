from ultralytics import YOLO
import cv2
import math
import time
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(data="african-wildlife.yaml", epochs=100)