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
    # model = YOLO("yolo11n.pt")
    # model.train(data=r"african-wildlife.yaml", epochs=10)


    model=YOLO(r"E:\share\code\ultralytics_qiyuan\ultralytics\runs\train\qiyuan\train_yolo11m_6404\weights\best.pt")
    print('class:', model.names)#class: {0: 'car', 1: 'bus', 2: 'truck', 3: 'person', 4: 'van', 5: 'motor', 6: 'tricycle', 7: 'ship', 8: 'bicycle', 9: 'plane'}