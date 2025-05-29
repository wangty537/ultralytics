import os
import cv2
import json
import glob
from ultralytics import YOLO

def main():
    # 加载训练好的模型
    model = YOLO("runs/detect/custom_train/weights/best.pt")
    
    # 从配置文件读取类别名称
    class_names = {
        0: "car", 1: "truck", 2: "bus", 3: "person", 4: "van",
        5: "motor", 6: "bicycle", 7: "ship", 8: "tricycle", 9: "plane"
    }
    
    # 测试集图像路径
    test_dir = "datasets/train_coco8/images/test"
    
    # 获取所有测试图像
    image_patterns = [
        os.path.join(test_dir, "*.jpg"),
        os.path.join(test_dir, "*.jpeg"),
        os.path.join(test_dir, "*.png")
    ]
    
    test_images = []
    for pattern in image_patterns:
        test_images.extend(glob.glob(pattern))
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 存储所有检测结果
    all_detections = []
    detection_id = 0
    
    # 遍历每张测试图像
    for img_path in test_images:
        print(f"正在处理: {os.path.basename(img_path)}")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        # 进行推理
        results = model.predict(img, verbose=False)
        
        # 处理检测结果
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # 转换每个检测框
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    # 转换为 [x, y, w, h] 格式
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    detection = {
                        "id": detection_id,
                        "category": class_names.get(class_ids[i], f"class_{class_ids[i]}"),
                        "image_name": os.path.basename(img_path),
                        "bbox": [x, y, w, h],
                        "score": float(confidences[i])
                    }
                    
                    all_detections.append(detection)
                    detection_id += 1
    
    # 保存结果到JSON文件
    output_file = "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)
    
    print(f"检测完成！")
    print(f"总共处理了 {len(test_images)} 张图像")
    print(f"总共检测到 {len(all_detections)} 个目标")
    print(f"结果已保存到: {output_file}")

def plot_results(img_path, dump_path, bbox, score, category):
    img = cv2.imread(img_path)
    x, y, w, h = bbox
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
    cv2.putText(img, f"{category}: {score:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(dump_path, img)

def convert_to_competation_format():
    js = 'test_results.json'
    with open(js, 'r') as f:
        data = json.load(f)

    js = 'pred.json'
    with open(js, 'r') as f:
        data_guide = json.load(f)

    categories_map = {
        "person": 1,
        "car": 2,
        "ship": 3,
        "plane": 4,
        "truck": 5,
        "van": 6,
        "bus": 7,
        "motor": 8,
        "bicycle": 9,
        "tricycle": 10,
    }

    test_results = {}
    test_results["images"] = data_guide["images"]
    test_results["categories"] = data_guide["categories"]
    test_results["annotations"] = []

    for i, item in enumerate(data):
        if i < 100:
            plot_results(f'./datasets/train_coco8/images/test/{item["image_name"]}', f'./pred_imgs/test_results_competation_{i}.jpg', item['bbox'], item['score'], item['category'])
        
        assert item['category'] in categories_map
        test_results["annotations"].append({
            "id": item['id'],
            "image_id": int(item['image_name'].split('.')[0].split('_')[1])+1,
            "category_id": categories_map[item['category']],
            "bbox": item['bbox'],
            "score": item['score']
        })
    
    with open('test_results_competation.json', 'w') as f:
        json.dump(test_results, f)

if __name__ == "__main__":
    # main()
    convert_to_competation_format()