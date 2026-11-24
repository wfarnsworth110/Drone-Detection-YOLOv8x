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
    "edge_verified.mp4",
    fourcc,
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

frame_count = 0
detection_count = 0
confidences = []
edge_densities = []
inference_times_ms = []
t_start = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.predict(frame, conf=0.3, verbose=False, device="mps")
    inference_times_ms.append(results[0].speed["inference"])

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]

        if crop.size > 0:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_crop, threshold1=80, threshold2=180)
            edge_density = cv2.countNonZero(edges) / (crop.shape[0] * crop.shape[1])

            detection_count += 1
            confidences.append(float(conf))
            edge_densities.append(float(edge_density))

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

total_elapsed_s = time.perf_counter() - t_start
cap.release()
out.release()

mean_inf_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
total_ms_per_frame = (total_elapsed_s * 1000.0 / frame_count) if frame_count else 0.0
fps = (frame_count / total_elapsed_s) if total_elapsed_s > 0 else 0.0

print("Exported edge_verified.mp4")
print("--- Edge Verification Summary ---")
print(f"Frames processed:      {frame_count}")
print(f"Total detections:      {detection_count} ({detection_count / max(1, frame_count):.2f}/frame)")
if edge_densities:
    print(f"Mean confidence:       {np.mean(confidences):.4f}")
    print(
        f"Canny edge density:    mean={np.mean(edge_densities):.4f}, "
        f"min={np.min(edge_densities):.4f}, max={np.max(edge_densities):.4f}"
    )
print(f"Model inference time:  {mean_inf_ms:.2f} ms/frame")
print(f"Total pipeline speed:  {total_ms_per_frame:.2f} ms/frame ({fps:.2f} FPS)")
