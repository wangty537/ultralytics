import json
import math


# 读取json文件，coco提交格式的json文件，统计类别和框的大小
# 定义 COCO 文件路径
coco_file_path = '/home/share11/code/ultralytics_qiyuan/ultralytics/results.json'

# 加载 COCO 文件
with open(coco_file_path, 'r') as f:
    coco_data = json.load(f)

# 获取类别信息
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# 初始化统计字典
category_stats = {}
for cat_id in categories.keys():
    category_stats[cat_id] = {
        'count': 0,
        'areas': [],
        'widths': [],
        'heights': []
    }

# 遍历标注信息进行统计
for ann in coco_data['annotations']:
    cat_id = ann['category_id']
    area = ann['area']
    bbox = ann['bbox']  # [x_min, y_min, width, height]
    width = bbox[2]
    height = bbox[3]
    category_stats[cat_id]['count'] += 1
    category_stats[cat_id]['areas'].append(area)
    category_stats[cat_id]['widths'].append(width)
    category_stats[cat_id]['heights'].append(height)

def mean(data):
    if not data:
        return 0
    return sum(data) / len(data)

def variance(data):
    if len(data) < 2:
        return 0
    avg = mean(data)
    return sum((x - avg) ** 2 for x in data) / len(data)

def std_dev(data):
    return math.sqrt(variance(data))

# 输出统计结果
print("每类目标数量及边界框大小统计：")
for cat_id, stats in category_stats.items():
    cat_name = categories[cat_id]
    count = stats['count']
    areas = stats['areas']
    widths = stats['widths']
    heights = stats['heights']

    # 计算面积统计信息
    if areas:
        avg_area = mean(areas)
        min_area = min(areas)
        max_area = max(areas)
    else:
        avg_area = 0
        min_area = 0
        max_area = 0

    # 计算宽度统计信息
    avg_width = mean(widths)
    var_width = variance(widths)
    std_width = std_dev(widths)

    # 计算高度统计信息
    avg_height = mean(heights)
    var_height = variance(heights)
    std_height = std_dev(heights)

    print(f"类别: {cat_name} (ID: {cat_id})")
    print(f"  目标数量: {count}")
    print(f"  平均边界框面积: {avg_area:.2f}")
    print(f"  最小边界框面积: {min_area:.2f}")
    print(f"  最大边界框面积: {max_area:.2f}")
    print(f"  平均边界框宽度: {avg_width:.2f}")
    print(f"  边界框宽度方差: {var_width:.2f}")
    print(f"  边界框宽度标准差: {std_width:.2f}")
    print(f"  平均边界框高度: {avg_height:.2f}")
    print(f"  边界框高度方差: {var_height:.2f}")
    print(f"  边界框高度标准差: {std_height:.2f}")
    print()
