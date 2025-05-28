import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import defaultdict
import tqdm
def analyze_dataset(label_dir, image_dir, output_dir):
    """
    分析目标检测数据集的特性和分布，并进行可视化。

    Args:
        label_dir (str): 包含 YOLO 格式标签文件的目录路径。
        image_dir (str): 包含图像文件的目录路径。
        output_dir (str): 保存分析结果和图表的目录路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_widths = []
    all_heights = []
    all_aspect_ratios = []
    all_areas = []
    class_counts = defaultdict(int)
    image_widths = []
    image_heights = []
    bbox_centers_x = []
    bbox_centers_y = []

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    print(f"开始分析 {len(label_files)} 个标签文件...")

    for label_file in tqdm(label_files):
        label_path = os.path.join(label_dir, label_file)
        image_name = os.path.splitext(label_file)[0] + '.jpg'  # 假设图像是.jpg格式
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            image_name = os.path.splitext(label_file)[0] + '.png' # 尝试.png格式
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                print(f"警告: 找不到图像文件 {image_name}，跳过 {label_file}")
                continue

        # 读取图像尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图像文件 {image_path}，跳过 {label_file}")
            continue
        img_h, img_w, _ = img.shape
        image_widths.append(img_w)
        image_heights.append(img_h)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:])

                    # 归一化坐标转换为像素坐标
                    bbox_w_abs = width * img_w
                    bbox_h_abs = height * img_h

                    all_widths.append(bbox_w_abs)
                    all_heights.append(bbox_h_abs)
                    if bbox_h_abs > 0:
                        all_aspect_ratios.append(bbox_w_abs / bbox_h_abs)
                    all_areas.append(bbox_w_abs * bbox_h_abs)
                    class_counts[class_id] += 1

                    bbox_centers_x.append(center_x * img_w)
                    bbox_centers_y.append(center_y * img_h)

    print("数据收集完成，开始生成图表...")

    # 1. 类别分布
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('目标类别分布')
    plt.xlabel('类别ID')
    plt.ylabel('数量')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()

    # 2. 边界框宽度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(all_widths, bins=50, kde=True)
    plt.title('边界框宽度分布 (像素)')
    plt.xlabel('宽度')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'bbox_width_distribution.png'))
    plt.close()

    # 3. 边界框高度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(all_heights, bins=50, kde=True)
    plt.title('边界框高度分布 (像素)')
    plt.xlabel('高度')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'bbox_height_distribution.png'))
    plt.close()

    # 4. 边界框宽高比分布
    plt.figure(figsize=(10, 6))
    sns.histplot(all_aspect_ratios, bins=50, kde=True)
    plt.title('边界框宽高比分布 (宽度/高度)')
    plt.xlabel('宽高比')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'bbox_aspect_ratio_distribution.png'))
    plt.close()

    # 5. 边界框面积分布
    plt.figure(figsize=(10, 6))
    sns.histplot(all_areas, bins=50, kde=True)
    plt.title('边界框面积分布 (像素^2)')
    plt.xlabel('面积')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'bbox_area_distribution.png'))
    plt.close()

    # 6. 边界框中心点分布
    if bbox_centers_x and bbox_centers_y:
        plt.figure(figsize=(10, 10))
        plt.scatter(bbox_centers_x, bbox_centers_y, alpha=0.1, s=1)
        plt.title('边界框中心点分布')
        plt.xlabel('X坐标 (像素)')
        plt.ylabel('Y坐标 (像素)')
        plt.gca().invert_yaxis()  # 图像Y轴通常是向下增长的
        plt.savefig(os.path.join(output_dir, 'bbox_center_distribution.png'))
        plt.close()

    # 7. 图像宽度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(image_widths, bins=20, kde=True)
    plt.title('图像宽度分布 (像素)')
    plt.xlabel('宽度')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'image_width_distribution.png'))
    plt.close()

    # 8. 图像高度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(image_heights, bins=20, kde=True)
    plt.title('图像高度分布 (像素)')
    plt.xlabel('高度')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'image_height_distribution.png'))
    plt.close()

    print(f"分析完成。所有图表已保存到 {output_dir} 目录。")

    # 打印一些统计信息
    print("\n--- 统计摘要 ---")
    print(f"总标签文件数: {len(label_files)}")
    print(f"总边界框数: {len(all_widths)}")
    print("类别统计:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"  类别 {cls_id}: {count} 个")

    if all_widths:
        print(f"边界框宽度 (像素) - 均值: {np.mean(all_widths):.2f}, 中位数: {np.median(all_widths):.2f}, 标准差: {np.std(all_widths):.2f}")
    if all_heights:
        print(f"边界框高度 (像素) - 均值: {np.mean(all_heights):.2f}, 中位数: {np.median(all_heights):.2f}, 标准差: {np.std(all_heights):.2f}")
    if all_aspect_ratios:
        print(f"边界框宽高比 - 均值: {np.mean(all_aspect_ratios):.2f}, 中位数: {np.median(all_aspect_ratios):.2f}, 标准差: {np.std(all_aspect_ratios):.2f}")
    if all_areas:
        print(f"边界框面积 (像素^2) - 均值: {np.mean(all_areas):.2f}, 中位数: {np.median(all_areas):.2f}, 标准差: {np.std(all_areas):.2f}")
    if image_widths:
        print(f"图像宽度 (像素) - 均值: {np.mean(image_widths):.2f}, 中位数: {np.median(image_widths):.2f}, 标准差: {np.std(image_widths):.2f}")
    if image_heights:
        print(f"图像高度 (像素) - 均值: {np.mean(image_heights):.2f}, 中位数: {np.median(image_heights):.2f}, 标准差: {np.std(image_heights):.2f}")


if __name__ == '__main__':
    # 请根据你的实际路径修改以下变量
    label_directory = r'F:\allcode\ultralytics\datasets\qiyuan\train\labels'
    image_directory = r'F:\allcode\ultralytics\datasets\qiyuan\train\images' # 假设图像在images目录下
    output_directory = r'F:\allcode\ultralytics\dataset_analysis_results'

    analyze_dataset(label_directory, image_directory, output_directory)
    
    """
    分析完成。所有图表已保存到 F:\allcode\ultralytics\dataset_analysis_results 目录。
    
    --- 统计摘要 ---
    总标签文件数: 30000
    总边界框数: 547695
    类别统计:
    类别 0: 302301 个
    类别 1: 20066 个
    类别 2: 8885 个
    类别 3: 97062 个
    类别 4: 11914 个
    类别 5: 9232 个
    类别 6: 28386 个
    类别 7: 1958 个
    类别 8: 56439 个
    类别 9: 11452 个
    边界框宽度 (像素) - 均值: 45.19, 中位数: 31.00, 标准差: 45.72
    边界框高度 (像素) - 均值: 45.54, 中位数: 36.00, 标准差: 38.42
    边界框宽高比 - 均值: 1.13, 中位数: 0.93, 标准差: 0.79
    边界框面积 (像素^2) - 均值: 3250.29, 中位数: 1140.00, 标准差: 8149.46
    图像宽度 (像素) - 均值: 766.49, 中位数: 800.00, 标准差: 114.76
    图像高度 (像素) - 均值: 744.16, 中位数: 742.00, 标准差: 114.42
    """