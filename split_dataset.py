from ultralytics.data.split import autosplit


if __name__ == "__main__":
    autosplit(r"F:\allcode\ultralytics\datasets\qiyuan\train\images", weights=(0.95, 0.05, 0.0), annotated_only=False)