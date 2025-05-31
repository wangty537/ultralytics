from re import L
from ultralytics import YOLO
import cv2
import math
import time
import os

if __name__ == "__main__":
    import torch
    import torchvision
    print(torch.__version__)
    print(torchvision.__version__)
    print(torch.cuda.is_available())
    #model = YOLO("yolo11n.pt")


    project = "runs/train/qiyuan"
    name = "train_yolo17s_640"
    model = YOLO("/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/ultralytics/cfg/models/unet/yolo17s.yaml")
    model.train(data=r"/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/ultralytics/cfg/datasets/qiyuan.yaml", 
                epochs=100,
                batch=2,
                workers=4,
                project=project,
                name=name,
                resume=False,
                device="2",
                )
                

    
    print('###################################################### Train Done ######################################################')
    # Customize validation settings
    metrics = model.val( 
        #data="gastrointestinal.yaml",
        imgsz=320, 
        # batch=4, 
        # conf=0.25,          # Confidence threshold for predictions模型预测的每个目标框（bounding box）的置信度分数（通常是目标存在的概率）必须大于该阈值，才会被保留下来作为有效检测结果
        # iou=0.6,            # iou 指的是非极大值抑制（NMS）中的 IoU 阈值，用于去除重叠度较高的冗余框。NMS 算法会对所有预测框按置信度排序，依次选出最大置信度的框，并去除与其 IoU 大于阈值的其他框。
        device="2",
        half=False,          # 使用半精度浮点数进行计算，减少内存占用和计算时间
        plots=True,         # 是否绘制验证结果图
        project=project,  # Project name for saving results
        name=name + "_val320",
        )
    print('###################################################### Val Done ######################################################')