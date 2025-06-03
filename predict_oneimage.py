
from ultralytics import YOLO
import os
import json
from PIL import Image
from tqdm import tqdm
import gc


if __name__ == "__main__":
    # 1. 加载模型
    model = YOLO("/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/runs/train/qiyuan/train_yolo11n_640/weights/best.pt")

    # 2. 获取图片列表
    img_dir = "/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/datasets/test/images"
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print("Found", len(img_files), "images in", img_dir)

    # 3. 预测图片
    results = model.predict(img_files[0], 
                save=False, 
                imgsz=640, 
                batch=1, 
                device=[2],
                verbose=False)  # 关闭详细输出，避免信息过多
    
    print(results[0].boxes.xywh, results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        result.save(filename="predict_oneimage_result.jpg")  # save to disk