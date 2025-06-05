from re import L
from ultralytics import YOLO
import cv2
import math
import time
from time import sleep

from load_repbyconv import load_pretrained_weights
if __name__ == "__main__":
    import torch
    import torchvision
    print(torch.__version__)
    print(torchvision.__version__)
    print(torch.cuda.is_available())

     # 加载新模型
    model = YOLO("yolo11m.yaml") # 修改Conv类为rep
    # 加载pretrain权重
    pretrain = torch.load("yolo11m.pt")

    print("Loading pretrain weights...")
    load_pretrained_weights(model.model.model, pretrain["model"].model)
    # 打印weights加载情况
    print("Pretrained weights loaded successfully.")

    # model = YOLO("yolomy11m.yaml") #yolomyrep11m 和 yolomy11m 完全一致，但是修改了Conv类
    # state = torch.load("/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/runs/detect/yolo11m/train640/weights/best.pt")


    project = "runs/detect/yolo11m_rep_adam_multiscale"
    #model = YOLO("/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/runs/train/qiyuan/train_yolo11l_640/weights/best.pt")
    model.train(data=r"/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/ultralytics/cfg/datasets/qiyuan.yaml", 
                epochs=100,
                # batch=4,
                # imgsz=640,
                # workers=4,
                # device=[0],
                # patience=10,
                optimizer="Adam",
                lr0=0.001,
                lrf=0.01,
                # close_mosaic=10,
                # # momentum=0.937,
                # # weight_decay=0.0005,
                # # warmup_epochs=3.0,
                # # warmup_momentum=0.8,
                # # warmup_bias_lr=0.1,
                # box=7.5,
                # cls=0.5,
                # dfl=1.5,
                multi_scale=True,
                project=project,
                name="train640",
                # resume=True,
                
                hsv_h=0.5,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=180,
                translate=0.1,
                scale=0.5,
                shear=60,
                perspective=0.0005,
                flipud=0.1,
                fliplr=0.5,
                bgr=0.1,
                mosaic=0.8,
                mixup=0.3,
                cutmix=0.3
                )
                

    
    print('###################################################### Train Done ######################################################')
    # Customize validation settings
    metrics = model.val( 
        #data="gastrointestinal.yaml",
        imgsz=320, 
        batch=4, 
        # conf=0.25,          # Confidence threshold for predictions模型预测的每个目标框（bounding box）的置信度分数（通常是目标存在的概率）必须大于该阈值，才会被保留下来作为有效检测结果
        # iou=0.6,            # iou 指的是非极大值抑制（NMS）中的 IoU 阈值，用于去除重叠度较高的冗余框。NMS 算法会对所有预测框按置信度排序，依次选出最大置信度的框，并去除与其 IoU 大于阈值的其他框。
        device=[1],
        half=False,          # 使用半精度浮点数进行计算，减少内存占用和计算时间
        plots=True,         # 是否绘制验证结果图
        project=project,  # Project name for saving results
        name="val320",
        )
    print('###################################################### Val Done ######################################################')
    # CUDA_VISIBLE_DEVICES=2 python train_qiyuan2.py