import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


def get_device() -> str:
    """Select the best available hardware accelerator."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_frame(video_path: str, frame_idx: int):
    """Read a single BGR frame from the target video."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path.resolve()}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path.resolve()}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"frame_idx {frame_idx} out of range (0..{total_frames - 1})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def apply_clahe_lab(bgr_img, clip_limit: float = 2.5, tile_size: int = 8):
    """Enhance local contrast on the LAB luminance channel while preserving color."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enh = clahe.apply(l_chan)
    merged = cv2.merge((l_enh, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def summarize_detections(label: str, result) -> None:
    """Print detection count and confidence scores for side-by-side logging."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print(f"{label}: 0 detections")
        return
    confs = [f"{c:.4f}" for c in boxes.conf.cpu().tolist()]
    print(f"{label}: {len(boxes)} detection(s) | conf={confs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard vs. CLAHE-Enhanced YOLOv8x Detection")
    parser.add_argument("--weights", default="best.pt", help="Path to trained YOLOv8x weights")
    parser.add_argument(
        "--video",
        default="test/pexels-joseph-redfield-8459631 (1080p).mp4",
        help="Path to input video",
    )
    parser.add_argument("--frame", type=int, default=450, help="Frame index to extract (0..551)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--clip", type=float, default=2.5, help="CLAHE clip limit")
    parser.add_argument("--out", default="detection_comparison.png", help="Output figure filename")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path.resolve()}")

    device = get_device()
    model = YOLO(str(weights_path))

    # Extract still frame and build contrast-enhanced copy
    orig = extract_frame(args.video, args.frame)
    enhanced = apply_clahe_lab(orig, clip_limit=args.clip)

    # Run YOLOv8x inference on both versions
    pred_orig = model.predict(orig, conf=args.conf, device=device, verbose=False)[0]
    pred_enh = model.predict(enhanced, conf=args.conf, device=device, verbose=False)[0]

    summarize_detections(f"Frame {args.frame} Standard YOLOv8x", pred_orig)
    summarize_detections(f"Frame {args.frame} CLAHE-Enhanced", pred_enh)

    # Render annotated frames side by side
    res_orig = pred_orig.plot()
    res_enh = pred_enh.plot()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(res_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Standard YOLOv8x (Frame {args.frame})")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(res_enh, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"CLAHE LAB Contrast-Enhanced (Frame {args.frame})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
