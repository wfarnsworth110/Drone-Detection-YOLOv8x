from collections import defaultdict
import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")

# Load video
video_path = "test/pexels-joseph-redfield-8459631 (1080p).mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "trajectory_tracked.mp4",
    fourcc,
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

track_history = defaultdict(list)
unique_ids = set()
tracked_detections = 0
confidences = []
step_distances_px = []
inference_times_ms = []
frame_count = 0
t_start = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False, device="mps")
    inference_times_ms.append(results[0].speed["inference"])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            x, y, w, h = box
            unique_ids.add(track_id)
            tracked_detections += 1
            confidences.append(float(conf))

            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) >= 2:
                dist = math.hypot(track[-1][0] - track[-2][0], track[-1][1] - track[-2][1])
                step_distances_px.append(dist)
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

total_elapsed_s = time.perf_counter() - t_start
cap.release()
out.release()

mean_inf_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
total_ms_per_frame = (total_elapsed_s * 1000.0 / frame_count) if frame_count else 0.0
fps = (frame_count / total_elapsed_s) if total_elapsed_s > 0 else 0.0
id_switches = max(0, len(unique_ids) - 1)

print("Exported trajectory_tracked.mp4")
print("--- ByteTrack Trajectory Summary ---")
print(f"Frames processed:      {frame_count}")
print(f"Tracked detections:    {tracked_detections} ({tracked_detections / max(1, frame_count):.2f}/frame)")
print(f"Unique track IDs:      {len(unique_ids)} (ID switches: {id_switches})")
if confidences:
    print(f"Mean confidence:       {np.mean(confidences):.4f}")
if step_distances_px:
    print(f"Mean step displacement:{np.mean(step_distances_px):.2f} px/frame")
    print(f"Total path length:     {np.sum(step_distances_px):.1f} px")
print(f"Model inference time:  {mean_inf_ms:.2f} ms/frame")
print(f"Total pipeline speed:  {total_ms_per_frame:.2f} ms/frame ({fps:.2f} FPS)")
