from ultralytics import YOLO
import pandas as pd
import os

def main():

    model = YOLO("results/yolo11_training/weights/best.pt")

    metrics = model.val(
        data="datasets/merged_yolo/dataset.yaml",
        split="test",
        conf=0.25,
        iou=0.5
    )

    results = {
        "mAP50": metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "F1_Score":
            2 * (metrics.box.mp * metrics.box.mr) /
            (metrics.box.mp + metrics.box.mr + 1e-9)
    }

    df = pd.DataFrame([results])

    os.makedirs("results/metrics", exist_ok=True)
    df.to_csv("results/metrics/yolo11_metrics.csv", index=False)

    print(df)

if __name__ == "__main__":
    main()