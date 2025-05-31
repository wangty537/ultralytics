import xml.etree.ElementTree as ET
import os


def convert_coordinates(size, box):
    """将 (xmin, xmax, ymin, ymax) 转换为归一化的 (x_center, y_center, width, height)"""
    box[0] = max(0, min(box[0], size[0]))  # 确保 xmin 不小于0且不大于图像宽度
    box[1] = max(0, min(box[1], size[0]))  # 确保 xmax 不小于0且不大于图像宽度
    box[2] = max(0, min(box[2], size[1]))  # 确保 ymin 不小于0且不大于图像高度
    box[3] = max(0, min(box[3], size[1]))  # 确保 ymax 不小于0且不大于图像高度
    

    dw = 1.0 / size[0]  # 图像宽度
    dh = 1.0 / size[1]  # 图像高度
    x_center = (box[0] + box[1]) / 2.0 * dw
    y_center = (box[2] + box[3]) / 2.0 * dh
    width = (box[1] - box[0]) * dw
    height = (box[3] - box[2]) * dh
    
    return (x_center, y_center, width, height)

def convert_xml_to_txt(xml_dir, txt_dir):
    """将指定目录下的 XML 文件转换为 TXT 文件"""
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    class_to_id_map = {}  # 用于存储类别名称到ID的映射
    next_class_id = 0     #下一个可用的类别ID

    # 获取所有XML文件列表，确保处理顺序一致性（可选，但有助于调试）
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')])

    for xml_file in xml_files:
        # if not xml_file.endswith('.xml'): # 已被上面的列表推导式过滤
        #     continue
        
        xml_path = os.path.join(xml_dir, xml_file)
        txt_file_name = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(txt_dir, txt_file_name)
        
        print(f"Processing {xml_path} -> {txt_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size_element = root.find('size')
            if size_element is None:
                print(f"Skipping {xml_file}: 'size' element not found.")
                continue
            
            image_width = int(size_element.find('width').text)
            image_height = int(size_element.find('height').text)
            
            if image_width == 0 or image_height == 0:
                print(f"Skipping {xml_file}: image_width or image_height is zero.")
                continue

            with open(txt_path, 'w') as txt_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    
                    if class_name not in class_to_id_map:
                        class_to_id_map[class_name] = next_class_id
                        next_class_id += 1
                    class_id = class_to_id_map[class_name]
                    
                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        print(f"Skipping object in {xml_file}: 'bndbox' element not found.")
                        continue
                        
                    xmin = float(bndbox.find('xmin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymin = float(bndbox.find('ymin').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # 确保 xmin < xmax 和 ymin < ymax
                    if xmin >= xmax or ymin >= ymax:
                        print(f"Warning: Invalid bounding box coordinates (xmin >= xmax or ymin >= ymax) in {xml_file}. Skipping object.")
                        continue

                    bb = (xmin, xmax, ymin, ymax)
                    converted_box = convert_coordinates((image_width, image_height), bb)
                    
                    txt_file.write(f"{class_id} {converted_box[0]:.6f} {converted_box[1]:.6f} {converted_box[2]:.6f} {converted_box[3]:.6f}\n")
            print(f"Successfully converted {xml_file} to {txt_file_name}")

        except ET.ParseError:
            print(f"Error parsing XML file: {xml_path}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {xml_path}: {e}. Skipping.")
    return class_to_id_map

if __name__ == '__main__':
    # 设置 XML 文件所在的目录和 TXT 文件要保存的目录
    # 注意：用户提供的路径是 F:\allcode\ultralytics\dataset\train\labels
    # 我们将 TXT 文件也保存在同一个目录下，或者可以指定一个新的目录
    xml_input_dir = r'F:\allcode\ultralytics\datasets\qiyuan\train\labels'
    txt_output_dir = xml_input_dir # 或者指定一个新的输出目录，例如 r'F:\allcode\ultralytics\dataset\train\labels_txt'
    
    # 如果输出目录与输入目录相同，并且希望在原XML旁边生成TXT，这是可以的。
    # 如果希望输出到新的子目录，例如 'labels_yolo'，可以这样设置：
    # txt_output_dir = os.path.join(os.path.dirname(xml_input_dir), 'labels_yolo')

    print(f"Starting conversion from XML in '{xml_input_dir}' to TXT in '{txt_output_dir}'...")
    class_map = convert_xml_to_txt(xml_input_dir, txt_output_dir)
    print("Conversion process finished.")
    if class_map:
        print("\nGenerated Class to ID Mapping:")
        for name, id_val in class_map.items():
            print(f"  '{name}': {id_val}")
            
#     Generated Class to ID Mapping:
#   'car': 0
#   'bus': 1
#   'truck': 2
#   'person': 3
#   'van': 4
#   'motor': 5
#   'tricycle': 6
#   'ship': 7
#   'bicycle': 8
#   'plane': 9

# testid = {0:2,1:7,2:5,3:1,4:6,5:8,6:10,7:3,8:9,9:4}
        # test
        # {
        #     "id": 1,
        #     "name": "person"
        # },
        # {
        #     "id": 2,
        #     "name": "car"
        # },
        # {
        #     "id": 3,
        #     "name": "ship"
        # },
        # {
        #     "id": 4,
        #     "name": "plane"
        # },
        # {
        #     "id": 5,
        #     "name": "truck"
        # },
        # {
        #     "id": 6,
        #     "name": "van"
        # },
        # {
        #     "id": 7,
        #     "name": "bus"
        # },
        # {
        #     "id": 8,
        #     "name": "motor"
        # },
        # {
        #     "id": 9,
        #     "name": "bicycle"
        # },
        # {
        #     "id": 10,
        #     "name": "tricycle"
        # }