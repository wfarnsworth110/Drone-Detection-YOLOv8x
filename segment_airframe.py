import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("pexels-joseph-redfield-8459631(1080p).mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "segmented_drone.mp4",
    fourcc,
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  results = model.predict(frame, conf=0.3, verbose=False, device="mps")
  for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
    if crop.size > 0:
      gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
      _, mask = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      green_mask = np.zeros_like(crop)
      green_mask[mask == 255] = [0, 255, 0]
      frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)] = cv2.addWeighted(
          crop, 0.7, green_mask, 0.3, 0
      )
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

  out.write(frame)

cap.release()
out.release()
print("Exported segmented_drone.mp4")
