
import cv2
from ultralytics import YOLO
import os
import json
from PIL import Image
from tqdm import tqdm
import gc

# 1. 加载模型
#model = YOLO("/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/runs/train/qiyuan/train_yolo11l_640/weights/best.pt")
model=YOLO(r"E:\share\code\ultralytics_qiyuan\ultralytics\runs\train\qiyuan\train_yolo11m_6404\weights\best.pt")

# 2. 获取图片列表
img_dir = "/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/datasets/test/images"
img_dir=r"E:\share\code\ultralytics_qiyuan\ultralytics\datasets\test\images"
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]#[:100]

print("Found", len(img_files), "images in", img_dir)

# 3. 预测并构建COCO格式
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}
# 实际测试输出的要求的类型id与名字的对应关系如下
test_names = [ {
            "id": 1,
            "name": "person"
        },
        {
            "id": 2,
            "name": "car"
        },
        {
            "id": 3,
            "name": "ship"
        },
        {
            "id": 4,
            "name": "plane"
        },
        {
            "id": 5,
            "name": "truck"
        },
        {
            "id": 6,
            "name": "van"
        },
        {
            "id": 7,
            "name": "bus"
        },
        {
            "id": 8,
            "name": "motor"
        },
        {
            "id": 9,
            "name": "bicycle"
        },
        {
            "id": 10,
            "name": "tricycle"
        }
]

# 直接使用 test_names 填充 coco["categories"]
coco["categories"] = test_names
# 创建类别名称到 test_names 中 id 的映射
name_to_id = {item["name"]: item["id"] for item in test_names}

# 假设类别名与模型训练时一致
# class_names = model.names  # dict: {0: 'person', 1: 'car', ...}
# for i, name in class_names.items():
#     coco["categories"].append({"id": i, "name": name})

ann_id = 1
img_id = 0
# 使用 tqdm 显示进度条
for file in tqdm(img_files):
    img_id += 1
    img = cv2.imread(file)
    height, width, _ = img.shape
    coco["images"].append({
        "id": img_id,
        "file_name": os.path.basename(file),
        "height": height,
        "width": width
    })
        
    results = model.predict(img, 
                            #   save=False, 
                            #   imgsz=640, 
                            #   batch=batch_size, 
                            #device=[2],
                            verbose=False)  # 关闭详细输出，避免信息过多

    
    result = results[0]

    # annotations字段
    for box, score, cls in zip(result.boxes.xywh, result.boxes.conf, result.boxes.cls):
        #print(img_id, box, score, cls)
        x, y, w, h = box.tolist()
        class_name = model.names[int(cls)]
        category_id = name_to_id[class_name]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [x - w/2, y - h/2, w, h],  # COCO格式为[x_min, y_min, w, h]
            "score": float(score),
            "area": float(w * h),
            "iscrowd": 0
        })
        ann_id += 1

print("total bbox number:", ann_id)

# 5. 保存为json
with open("resultsyolo11m.json", "w") as f:
    json.dump(coco, f)
