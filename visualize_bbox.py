import os
import random
import cv2
import numpy as np

def denormalize_bbox(img_width, img_height, x_center_norm, y_center_norm, width_norm, height_norm):
    """将归一化的YOLO格式边界框转换为 (xmin, ymin, xmax, ymax) 格式"""
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height

    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)

    return xmin, ymin, xmax, ymax

def visualize_random_image_with_bbox(image_dir, label_dir, class_names=None, idx=None):
    """随机选择一张图片及其对应的TXT标签，并在图片上绘制边界框进行可视化"""
    # 定义一个颜色列表，用于不同类别的边界框
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋红色
        (0, 255, 255),  # 黄色
        (128, 0, 0),    # 深红色
        (0, 128, 0),    # 深绿色
        (0, 0, 128),    # 深蓝色
        (128, 128, 0),  # 橄榄色
        (128, 0, 128),  # 紫色
        (0, 128, 128),  # 蓝绿色
        (255, 165, 0),  # 橙色
        (255, 215, 0),  # 金色
        (165, 42, 42),   # 棕色
    ]
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    if idx is not None:
        if 0 <= idx < len(image_files):
            random_image_name = image_files[idx]
        else:
            print(f"Invalid index: {idx}. Index should be between 0 and {len(image_files) - 1}.")
            return
    else:
        random_image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image_name)

    base_name, _ = os.path.splitext(random_image_name)
    label_file_name = base_name + '.txt'
    label_path = os.path.join(label_dir, label_file_name)

    if not os.path.exists(label_path):
        print(f"Label file not found for {random_image_name} at {label_path}")
        return

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    img_height, img_width, _ = image.shape
    print(f"Visualizing: {image_path}")
    print(f"Image dimensions: Width={img_width}, Height={img_height}")

    # 初始化类别计数器
    class_counts = {}

    # 读取TXT标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            # 更新类别计数
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # 将归一化坐标转回原始坐标
            xmin, ymin, xmax, ymax = denormalize_bbox(img_width, img_height, x_center_norm, y_center_norm, width_norm, height_norm)
            
            # 绘制边界框
            color = colors[class_id % len(colors)] # 根据类别ID选择颜色，如果类别数超过颜色数则循环使用
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

            # 可选：绘制类别标签
            label_text = str(class_id)
            if class_names and 0 <= class_id < len(class_names):
                label_text = class_names[class_id]
            
            cv2.putText(image, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"  Label: {label_text}, BBox (denormalized): xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
        else:
            print(f"Skipping invalid line in {label_file_name}: {line.strip()}")
    print("total bbox: ", len(lines))

    # 打印每个类别的目标数量
    print("\nObject counts per class:")
    for cid, count in sorted(class_counts.items()):
        class_name_str = str(cid)
        if class_names and 0 <= cid < len(class_names):
            class_name_str = class_names[cid]
        print(f"  Class {class_name_str}: {count}")

    # 显示图片
    cv2.imshow(f"Image with BBoxes - {random_image_name}", image)
    cv2.waitKey(0)  # 等待按键后关闭
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # # 用户指定的目录
    # base_data_dir = r'E:\share\code\ultralytics_qiyuan\ultralytics\datasets\train'
    # image_dir = os.path.join(base_data_dir, 'images')
    # label_dir = os.path.join(base_data_dir, 'labels')

    # # 假设的类别名称，如果您的 convert_xml_to_txt.py 脚本输出了类别映射，可以在这里使用
    # # 例如: class_names = ['car', 'person', 'cat'] # 索引对应 convert_xml_to_txt.py 生成的 class_id
    # # 如果没有类别名称，将只显示类别ID
    # # 您可以从 convert_xml_to_txt.py 脚本的输出中获取 class_map 并转换为列表
    # # 示例：class_map = {'cat': 0, 'dog': 1} -> class_names = ['cat', 'dog'] (需要确保顺序正确)
    # # 为了简单起见，这里暂时不使用 class_names，只显示ID
    # # class_names_example = ['class0', 'class1', 'class2'] # 替换为您的实际类别名称
    # class_map = {
    #     "person": 0,
    #     "car": 1,
    #     "ship": 2,
    #     "plane": 3,
    #     "truck": 4,
    #     "van": 5,
    #     "bus": 6,
    #     "motor": 7,
    #     "bicycle": 8,
    #     "tricycle": 9
    # }
    # class_names = list(class_map.keys())
    # print(f"Class Names: {class_names}")
    # if not os.path.exists(image_dir):
    #     print(f"Image directory not found: {image_dir}")
    # elif not os.path.exists(label_dir):
    #     print(f"Label directory not found: {label_dir}")
    # else:
    #     visualize_random_image_with_bbox(image_dir, label_dir, class_names=class_names, idx=None)
        
    
    # visdrone
    # 用户指定的目录
    base_data_dir = r'E:\share\code\ultralytics_qiyuan\ultralytics\datasets2\VisDrone2019-DET-train'
    image_dir = os.path.join(base_data_dir, 'images')
    label_dir = os.path.join(base_data_dir, 'labels')

    # 假设的类别名称，如果您的 convert_xml_to_txt.py 脚本输出了类别映射，可以在这里使用
    # 例如: class_names = ['car', 'person', 'cat'] # 索引对应 convert_xml_to_txt.py 生成的 class_id
    # 如果没有类别名称，将只显示类别ID
    # 您可以从 convert_xml_to_txt.py 脚本的输出中获取 class_map 并转换为列表
    # 示例：class_map = {'cat': 0, 'dog': 1} -> class_names = ['cat', 'dog'] (需要确保顺序正确)
    # 为了简单起见，这里暂时不使用 class_names，只显示ID
    # class_names_example = ['class0', 'class1', 'class2'] # 替换为您的实际类别名称

    names = {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
    class_map = {v: k for k, v in names.items()}

    class_names = list(class_map.keys())
    print(f"Class Names: {class_names}")
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
    elif not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
    else:
        visualize_random_image_with_bbox(image_dir, label_dir, class_names=class_names, idx=None)
    
    