"""
Baseline YOLOv8x drone detection & tracking script adapted for local execution.
"""

import argparse
from pathlib import Path
import warnings

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO, checks

warnings.filterwarnings("ignore")


def get_device() -> str:
    """Select the best available accelerator."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_train(data_yaml: str, epochs: int, batch: int, imgsz: int, device: str) -> Path:
    """Train YOLOv8x baseline from pretrained weights."""
    model = YOLO("yolov8x.pt")
    results = model.train(
        batch=batch,
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
    )
    save_dir = Path(results.save_dir)
    results_plot = save_dir / "results.png"
    if results_plot.exists():
        img = mpimg.imread(results_plot)
        plt.figure(figsize=(16, 9), dpi=150)
        plt.title("Training Results")
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    return save_dir / "weights" / "best.pt"


def run_track(weights_path: str, video_path: str, conf: float, iou: float, show: bool, device: str) -> None:
    """Run baseline YOLOv8x tracking on the target video using streaming mode."""
    weights = Path(weights_path)
    video = Path(video_path)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights.resolve()}")
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video.resolve()}")

    model = YOLO(str(weights))
    results = model.track(
        source=str(video),
        conf=conf,
        iou=iou,
        device=device,
        show=show,
        save=True,
        stream=True,
    )

    frame_count = 0
    total_detections = 0
    for r in results:
        frame_count += 1
        if r.boxes is not None:
            total_detections += len(r.boxes)

    print(f"Baseline tracking complete: {frame_count} frames processed, {total_detections} total detections.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8x Drone Detection Baseline")
    parser.add_argument("--mode", choices=["track", "train", "both"], default="track")
    parser.add_argument("--weights", default="results/best.pt", help="Path to trained best.pt weights")
    parser.add_argument("--video", default="content/pexels-joseph-redfield-8459631(1080p).mp4")
    parser.add_argument("--data", default="coco.yaml")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window")
    args = parser.parse_args()

    checks()
    device = get_device()
    print(f"Using device: {device}")

    weights_path = args.weights
    if args.mode in ("train", "both"):
        weights_path = str(run_train(args.data, args.epochs, args.batch, args.imgsz, device))

    if args.mode in ("track", "both"):
        run_track(weights_path, args.video, args.conf, args.iou, args.show, device)


if __name__ == "__main__":
    main()
