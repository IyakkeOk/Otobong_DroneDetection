from ultralytics import YOLO
import time
import cv2

model = YOLO("results/yolo11_training/weights/best.pt")

img = cv2.imread("sample.jpg")

runs = 100
start = time.time()

for _ in range(runs):
    model.predict(img, verbose=False)

end = time.time()

fps = runs / (end - start)
print("FPS:", fps)