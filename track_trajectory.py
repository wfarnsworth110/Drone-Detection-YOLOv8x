from collections import defaultdict
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("pexels-joseph-redfield-8459631(1080p).mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "trajectory_tracked.mp4",
    fourcc,
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

track_history = defaultdict(lambda: [])

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False, device="mps")
  if results[0].boxes.id is not None:
    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    for box, track_id in zip(boxes, track_ids):
      x, y, w, h = box
      track = track_history[track_id]
      track.append((float(x), float(y)))
      if len(track) > 30:
        track.pop(0)

      # Draw motion trajectory
      for i in range(1, len(track)):
        cv2.line(
            frame,
            (int(track[i - 1][0]), int(track[i - 1][1])),
            (int(track[i][0]), int(track[i][1])),
            (0, 0, 255),
            2,
        )

      # Draw bounding box and persistent ID
      x1, y1 = int(x - w / 2), int(y - h / 2)
      cv2.rectangle(frame, (x1, y1), (int(x + w / 2), int(y + h / 2)), (0, 255, 255), 2)
      cv2.putText(
          frame,
          f"UAV ID: {track_id}",
          (x1, max(15, y1 - 8)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 255),
          2,
      )

  out.write(frame)

cap.release()
out.release()
print("Exported trajectory_tracked.mp4")
