from ultralytics import YOLO
import os

def main():

    model = YOLO("yolo11n.pt")   # nano version first

    results = model.train(
        data="datasets/merged_yolo/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        patience=20,
        project="results",
        name="yolo11_training",
        save=True,
        save_period=10,
        pretrained=True,
        cache=True
    )

    print("Training Complete")

if __name__ == "__main__":
    main()