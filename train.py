from ultralytics import YOLO
import cv2
import math
import time
if __name__ == "__main__":
    import torch
    import torchvision
    print(torch.__version__)
    print(torchvision.__version__)
    print(torch.cuda.is_available())
    model = YOLO("yolo11n.pt")
    model.train(data=r"african-wildlife.yaml", epochs=10)