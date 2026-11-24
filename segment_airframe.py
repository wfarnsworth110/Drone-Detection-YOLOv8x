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
    "segmented_drone.mp4",
    fourcc,
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

frame_count = 0
detection_count = 0
confidences = []
otsu_thresholds = []
mask_coverages_pct = []
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
            otsu_t, mask = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            fg_pct = 100.0 * cv2.countNonZero(mask) / mask.size
            detection_count += 1
            confidences.append(float(conf))
            otsu_thresholds.append(float(otsu_t))
            mask_coverages_pct.append(float(fg_pct))

            green_mask = np.zeros_like(crop)
            green_mask[mask == 255] = [0, 255, 0]
            frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)] = cv2.addWeighted(
                crop, 0.7, green_mask, 0.3, 0
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)

total_elapsed_s = time.perf_counter() - t_start
cap.release()
out.release()

mean_inf_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
total_ms_per_frame = (total_elapsed_s * 1000.0 / frame_count) if frame_count else 0.0
fps = (frame_count / total_elapsed_s) if total_elapsed_s > 0 else 0.0

print("Exported segmented_drone.mp4")
print("--- Airframe Segmentation Summary ---")
print(f"Frames processed:      {frame_count}")
print(f"Total detections:      {detection_count} ({detection_count / max(1, frame_count):.2f}/frame)")
if otsu_thresholds:
    print(f"Mean confidence:       {np.mean(confidences):.4f}")
    print(f"Mean Otsu threshold:   {np.mean(otsu_thresholds):.2f} (range: {np.min(otsu_thresholds):.0f}-{np.max(otsu_thresholds):.0f})")
    print(f"Mean mask coverage:    {np.mean(mask_coverages_pct):.2f}% of bbox area")
print(f"Model inference time:  {mean_inf_ms:.2f} ms/frame")
print(f"Total pipeline speed:  {total_ms_per_frame:.2f} ms/frame ({fps:.2f} FPS)")
