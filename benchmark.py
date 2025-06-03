from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="/home/redpine/share11/code/ultralytics_qiyuan/ultralytics/runs/train/qiyuan/train_yolo11n_640/weights/best.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

# Benchmark specific export format
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")