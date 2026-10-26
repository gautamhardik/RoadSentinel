"""
Run YOLO26 inference with custom-trained weights.

Supports three source types:
  - Single image file
  - Directory of images (batch prediction)
  - Video file
"""

import argparse
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26 inference on images, directories, or videos.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to custom YOLO26 weights (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, directory of images, or video file.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS when using one-to-many head (default: 0.7).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. 'cpu', '0', 'mps' (default: auto).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/predict",
        help="Directory to save annotated results (default: runs/predict).",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as .txt labels.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write detection results to a JSON file in --save-dir.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated results on screen.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated outputs to disk.",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        default=True,
        help="Use the NMS-free one-to-one head (default: True).",
    )
    parser.add_argument(
        "--no-end2end",
        dest="end2end",
        action="store_false",
        help="Use the one-to-many head (requires NMS).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for directory prediction (default: 16).",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Video frame stride — process every Nth frame (default: 1).",
    )
    return parser.parse_args()


def detect_source_type(source: Path) -> str:
    """Return 'image', 'directory', or 'video' based on the source path."""
    if source.is_dir():
        return "directory"
    ext = source.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    sys.exit(
        f"Error: unsupported source type '{ext}'. "
        f"Supported images: {IMAGE_EXTENSIONS}, videos: {VIDEO_EXTENSIONS}, "
        "or pass a directory."
    )


def extract_detections(result) -> list[dict]:
    """Pull structured detection dicts from a single Results object."""
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
        })
    return detections


def build_predict_kwargs(args: argparse.Namespace) -> dict:
    """Build the kwargs dict passed to model.predict()."""
    kwargs: dict = {
        "imgsz": args.img_size,
        "conf": args.conf,
        "iou": args.iou,
        "save": not args.no_save,
        "save_txt": args.save_txt,
        "project": str(Path(args.save_dir).parent),
        "name": Path(args.save_dir).name,
        "end2end": args.end2end,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    return kwargs


def run_single_image(model: YOLO, source: Path, args: argparse.Namespace) -> None:
    """Run inference on a single image and print results."""
    kwargs = build_predict_kwargs(args)
    print(f"Running inference on {source} ...")

    results = model.predict(source=str(source), **kwargs)
    result = results[0]
    detections = extract_detections(result)

    print(f"\nDetected {len(detections)} object(s):")
    for det in detections:
        print(
            f"  [{det['class_name']}] conf={det['confidence']:.2f}  "
            f"bbox={det['bbox_xyxy']}"
        )

    if args.show:
        result.show()

    if args.save_json:
        _write_json(args.save_dir, {"images": [{
            "file": source.name,
            "detections": detections,
        }]})


def run_directory(model: YOLO, source: Path, args: argparse.Namespace) -> None:
    """Run batch inference on all images in a directory."""
    image_files = sorted(
        p for p in source.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        sys.exit(f"Error: no images found in {source}")

    print(f"Found {len(image_files)} image(s) in {source}")
    kwargs = build_predict_kwargs(args)
    kwargs["stream"] = True
    kwargs["batch"] = args.batch_size

    all_results: list[dict] = []
    total_detections = 0
    t0 = time.perf_counter()

    for result in model.predict(source=str(source), **kwargs):
        src_name = Path(result.path).name
        detections = extract_detections(result)
        total_detections += len(detections)
        all_results.append({"file": src_name, "detections": detections})

        print(f"  {src_name}: {len(detections)} detection(s)")

    elapsed = time.perf_counter() - t0
    fps = len(image_files) / elapsed if elapsed > 0 else 0.0
    print(
        f"\nBatch complete: {len(image_files)} images, "
        f"{total_detections} total detections, "
        f"{elapsed:.1f}s ({fps:.1f} img/s)"
    )

    if args.save_json:
        _write_json(args.save_dir, {"images": all_results})

    if not args.no_save:
        print(f"Annotated images saved to {args.save_dir}/")


def run_video(model: YOLO, source: Path, args: argparse.Namespace) -> None:
    """Run inference on a video file, frame-by-frame with streaming."""
    print(f"Running inference on video {source} (stride={args.vid_stride}) ...")
    kwargs = build_predict_kwargs(args)
    kwargs["stream"] = True
    kwargs["vid_stride"] = args.vid_stride

    all_frames: list[dict] = []
    frame_idx = 0
    total_detections = 0
    t0 = time.perf_counter()

    for result in model.predict(source=str(source), **kwargs):
        detections = extract_detections(result)
        total_detections += len(detections)
        all_frames.append({
            "frame": frame_idx,
            "detections": detections,
        })

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}: {len(detections)} detection(s)")
        frame_idx += 1

    elapsed = time.perf_counter() - t0
    fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(
        f"\nVideo complete: {frame_idx} frames processed, "
        f"{total_detections} total detections, "
        f"{elapsed:.1f}s ({fps:.1f} FPS)"
    )

    if args.save_json:
        _write_json(args.save_dir, {
            "video": source.name,
            "total_frames": frame_idx,
            "frames": all_frames,
        })

    if not args.no_save:
        print(f"Annotated video saved to {args.save_dir}/")


def _write_json(save_dir: str, data: dict) -> None:
    """Write results dict to a JSON file inside save_dir."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    json_path.write_text(json.dumps(data, indent=2))
    print(f"Results JSON saved to {json_path}")


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)

    if not weights_path.exists():
        sys.exit(f"Error: weights file not found: {weights_path}")
    if not source_path.exists():
        sys.exit(f"Error: source not found: {source_path}")

    print(f"Loading model from {weights_path} ...")
    model = YOLO(str(weights_path))

    source_type = detect_source_type(source_path)
    print(f"Source type: {source_type}\n")

    if source_type == "image":
        run_single_image(model, source_path, args)
    elif source_type == "directory":
        run_directory(model, source_path, args)
    elif source_type == "video":
        run_video(model, source_path, args)


if __name__ == "__main__":
    main()
