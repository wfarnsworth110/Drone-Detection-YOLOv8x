import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("pexels-joseph-redfield-8459631(1080p).mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "edge_verified.mp4",
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
      edges = cv2.Canny(gray_crop, threshold1=80, threshold2=180)
      edge_density = cv2.countNonZero(edges) / (crop.shape[0] * crop.shape[1])

      # Overlay edge map HUD in the corner of the bounding box
      edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
      h_crop, w_crop = min(edges.shape[0], 60), min(edges.shape[1], 60)
      edge_thumb = cv2.resize(edge_bgr, (w_crop, h_crop))
      frame[max(0, y1) : max(0, y1) + h_crop, max(0, x1) : max(0, x1) + w_crop] = edge_thumb

      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
      cv2.putText(
          frame,
          f"Edges: {edge_density:.2f}",
          (x1, max(15, y1 - 8)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (255, 0, 0),
          1,
      )

  out.write(frame)

cap.release()
out.release()
print("Exported edge_verified.mp4")
