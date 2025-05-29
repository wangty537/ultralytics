import os
from ultralytics import YOLO
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import shutil

def train_and_validate():
    """训练模型并在训练集上进行推理以检测错误标注"""
    
    # Create a new YOLO model from scratch
    # model = YOLO("yolo11m.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11m.pt")

    print("开始训练模型...")
    # Train the model using custom dataset
    results = model.train(
        data="custom_dataset.yaml",  # 使用自定义数据集配置
        epochs=50,                   # 训练轮数
        imgsz=640,                   # 图像尺寸
        batch=8,                    # 批量大小
        device=0,                    # 使用GPU 0
        project="runs/detect",       # 项目目录
        name="custom_train",         # 实验名称
        save=True,                   # 保存检查点
        save_period=10,              # 每10个epoch保存一次
        patience=20,                 # 早停耐心值
        verbose=True                 # 详细输出
    )

    print("训练完成！开始在训练集上进行推理...")
    
    # Load the best trained model
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    trained_model = YOLO(str(best_model_path))
    
    # Evaluate the model's performance on the validation set
    print("开始验证...")
    val_results = trained_model.val()
    
    return results, val_results, trained_model

def save_validation_results(train_results, val_results, trained_model):
    """保存验证结果、可视化并创建结果文件夹"""
    
    # 获取mAP指标
    metrics = val_results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map75 = metrics.get('metrics/mAP50-95(B)', 0.0)  # 注意：这实际上是mAP50-95
    
    # 如果有单独的mAP75，使用它；否则使用mAP50-95作为近似
    map75_actual = metrics.get('metrics/mAP75(B)', map75)
    
    # 计算均值
    map_mean = (map50 + map75_actual) / 2
    
    # 生成时间戳和文件夹名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{map_mean:.4f}"
    results_dir = Path("./res") / folder_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"创建结果文件夹: {results_dir}")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP75/mAP50-95: {map75_actual:.4f}")
    print(f"均值: {map_mean:.4f}")
    
    # 1. 保存精度指标到JSON文件
    metrics_summary = {
        "timestamp": timestamp,
        "mAP50": float(map50),
        "mAP75": float(map75_actual),
        "mAP50-95": float(map75),
        "mean_mAP": float(map_mean),
        "all_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                       for k, v in metrics.items()}
    }
    
    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    visualize_validation_predictions(trained_model, results_dir)
    
    # 2. 保存详细的验证结果
    with open(results_dir / "validation_details.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("YOLO模型验证结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"mAP50: {map50:.4f}\n")
        f.write(f"mAP75/mAP50-95: {map75_actual:.4f}\n")
        f.write(f"均值: {map_mean:.4f}\n")
        f.write("\n详细指标:\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # 4. 复制训练结果
    if train_results and hasattr(train_results, 'save_dir'):
        train_dir = Path(train_results.save_dir)
        if train_dir.exists():
            # 复制训练曲线图
            plots_to_copy = ['results.png', 'confusion_matrix.png', 'P_curve.png', 
                           'R_curve.png', 'PR_curve.png', 'F1_curve.png']
            for plot_name in plots_to_copy:
                src_plot = train_dir / plot_name
                if src_plot.exists():
                    shutil.copy2(src_plot, results_dir / plot_name)
                    print(f"复制训练图表: {plot_name}")
            
            # 复制最佳模型权重
            best_weights = train_dir / 'weights' / 'best.pt'
            if best_weights.exists():
                shutil.copy2(best_weights, results_dir / 'best_model.pt')
                print("复制最佳模型权重")
    
    print(f"\n所有结果已保存到: {results_dir}")
    return results_dir

def visualize_validation_predictions(model, results_dir, max_images=20):
    """在验证集上进行推理并可视化预测结果"""
    
    # 读取数据集配置以获取验证集路径
    try:
        with open("custom_dataset.yaml", "r", encoding="utf-8") as f:
            import yaml
            dataset_config = yaml.safe_load(f)
            
        val_path = dataset_config.get('val', '')
        if not val_path:
            print("警告: 无法从配置文件中获取验证集路径")
            return
            
        val_images_path = Path(val_path)
        if not val_images_path.exists():
            print(f"警告: 验证集路径不存在: {val_images_path}")
            return
            
    except Exception as e:
        print(f"读取数据集配置文件失败: {e}")
        return
    
    # 创建预测结果保存目录
    predictions_dir = results_dir / "validation_predictions"
    predictions_dir.mkdir(exist_ok=True)
    
    # 获取验证集中的所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(val_images_path.glob(f"*{ext}")))
        image_files.extend(list(val_images_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print("警告: 验证集中没有找到图像文件")
        return
    
    # 随机选择部分图像进行可视化
    if len(image_files) > max_images:
        import random
        random.seed(42)  # 确保结果可重现
        image_files = random.sample(image_files, max_images)
    
    print(f"开始处理 {len(image_files)} 张验证图像...")
    
    # 类别名称映射（从dataset.py中的CLASS_MAPPING获取）
    class_names = {
        0: 'car', 1: 'truck', 2: 'bus', 3: 'person', 4: 'van',
        5: 'motor', 6: 'bicycle', 7: 'ship', 8: 'tricycle', 9: 'plane'
    }
    
    # 颜色映射
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 128)
    ]
    predictions_summary = []
    
    for i, img_path in enumerate(image_files):
        try:
            # 进行推理
            results = model(str(img_path), conf=0.25, iou=0.45)
            
            # 读取原始图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
                
            img_height, img_width = img.shape[:2]
            
            # 获取预测结果
            detections = results[0].boxes if results[0].boxes is not None else None
            
            predictions_info = {
                "image_name": img_path.name,
                "image_size": [img_width, img_height],
                "detections": []
            }
            
            if detections is not None and len(detections) > 0:
                # 绘制检测结果
                for j, (box, conf, cls) in enumerate(zip(detections.xyxy, detections.conf, detections.cls)):
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    confidence = float(conf.cpu().numpy())
                    class_id = int(cls.cpu().numpy())
                    
                    # 获取类别名称和颜色
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    color = colors[class_id % len(colors)]
                    
                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = max(y1 - 10, label_size[1] + 10)
                    
                    cv2.rectangle(img, (x1, label_y - label_size[1] - 10), 
                                (x1 + label_size[0], label_y), color, -1)
                    cv2.putText(img, label, (x1, label_y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 保存检测信息
                    predictions_info["detections"].append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
            
            predictions_summary.append(predictions_info)
            # 保存可视化结果
            output_path = predictions_dir / f"pred_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), img)
            
            if (i + 1) % 5 == 0:
                print(f"已处理 {i + 1}/{len(image_files)} 张图像")
                
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            continue


if __name__ == "__main__":
    print("开始YOLO训练和标注验证流程...")
    train_results, val_results, trained_model = train_and_validate()
    
    print("\n保存验证结果和生成可视化...")
    results_dir = save_validation_results(train_results, val_results, trained_model)
    
    print("流程完成！")
    print(f"查看结果: {results_dir}")