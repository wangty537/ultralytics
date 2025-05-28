from re import L
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
    model.train(data=r"qiyuan.yaml", 
                epochs=200,
                batch=16,
                imgsz=320,
                workers=8,
                device=0,
                patience=30,
                optimizer="Adam",
                lr0=0.001,
                lrf=0.01,
                close_mosaic=20,
                # momentum=0.937,
                # weight_decay=0.0005,
                # warmup_epochs=3.0,
                # warmup_momentum=0.8,
                # warmup_bias_lr=0.1,
                box=0.05,
                cls=0.5,
                dfl=1.5,
                multi_scale=False,
                project="runs/train/qiyuan",
                name="yolo11n_320",
                resume=False,
                
                hsv_h=0.1,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=90,
                translate=0.1,
                scale=0.5,
                # shear=0,
                # perspective=0.0,
                flipud=0.1,
                fliplr=0.5,
                bgr=0.1,
                mosaic=0.8,
                mixup=0.5,
                cutmix=0.5
                )
                
                
                
                
    
    
    print("########################### training finished ###########################")
    
    model.val()