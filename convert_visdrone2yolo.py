import os
from pathlib import Path
import cv2
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
def visdrone2yolo(dir):
    """Convert VisDrone annotations to YOLO format, creating label files with normalized bounding box coordinates."""


    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    (dir / "labels").mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / "annotations").glob("*.txt"), desc=f"Converting {dir}")
    # ind = 0
    for f in pbar:
        # ind += 1
        # print(f"Processing {ind} files")
        img_size = Image.open((dir / "images" / f.name).with_suffix(".jpg")).size

        # img = cv2.imread((dir / "images" / f.name).with_suffix(".jpg"))  # read image to get size
        # img_size = img.shape[:2][::-1]  # (height, width)
        lines = []
        with open(f, encoding="utf-8") as file:  # read annotation.txt
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] == "0":  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f"{os.sep}annotations{os.sep}", f"{os.sep}labels{os.sep}")[:-4]+'.dd', "w", encoding="utf-8") as fl:
                    fl.writelines(lines)  # write label.txt

import glob
import shutil

def dd2txt(dir):
    files = glob.glob(os.path.join(dir, "*.dd"))
    for file in files:
        new_file = file[:-3] + ".txt"
        shutil.move(file, new_file)
if __name__ == "__main__":
    # dir = Path(r'/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/datasets2')
    # dir = Path(r'E:\share\code\ultralytics_qiyuan\ultralytics\datasets2')
    # # Convert
    # for d in "VisDrone2019-DET-train",: #, "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev":
    #     visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels


    dd2txt(r'/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/datasets2/VisDrone2019-DET-train/labels')