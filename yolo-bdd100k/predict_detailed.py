"""
Detailed YOLO26 inference library with full result extraction.

Supports:
  - Single image prediction
  - Batch prediction from a directory of images
  - Video prediction (frame-by-frame with streaming)

All public functions return structured dicts so results can be consumed
programmatically, logged, or serialised to JSON.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights: str, device: str | None = None) -> YOLO:
    """Load a YOLO26 model from a custom weights file."""
    model = YOLO(weights)
    if device:
        model.to(device)
    return model


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _extract_detections(result) -> list[dict]:
    """Extract structured detection dicts from a single Results object."""
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
            "bbox_xywh": [round(v, 2) for v in box.xywh[0].tolist()],
        })
    return detections


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------

def predict_image(
    model: YOLO,
    image_path: str | Path,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
) -> dict:
    """
    Run inference on a single image.

    Returns:
        dict with image_path, image_shape, detections, annotated_image.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )
    result = results[0]
    return {
        "image_path": str(image_path),
        "image_shape": result.orig_shape,
        "detections": _extract_detections(result),
        "annotated_image": result.plot(),
    }


# ---------------------------------------------------------------------------
# Batch (directory of images)
# ---------------------------------------------------------------------------

def predict_directory(
    model: YOLO,
    directory: str | Path,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    batch_size: int = 16,
) -> dict:
    """
    Run batch inference on every image inside *directory*.

    Uses stream=True for memory efficiency.

    Returns:
        dict with directory, total_images, total_detections, elapsed_s,
        fps, and a per-image results list.
    """
    directory = Path(directory)
    image_files = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {directory}")

    per_image: list[dict] = []
    total_detections = 0
    t0 = time.perf_counter()

    for result in model.predict(
        source=str(directory),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch=batch_size,
        save=False,
        stream=True,
        verbose=False,
    ):
        dets = _extract_detections(result)
        total_detections += len(dets)
        per_image.append({
            "file": Path(result.path).name,
            "image_shape": result.orig_shape,
            "detections": dets,
            "annotated_image": result.plot(),
        })

    elapsed = time.perf_counter() - t0
    return {
        "directory": str(directory),
        "total_images": len(per_image),
        "total_detections": total_detections,
        "elapsed_s": round(elapsed, 2),
        "fps": round(len(per_image) / elapsed, 1) if elapsed > 0 else 0.0,
        "results": per_image,
    }


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def predict_video(
    model: YOLO,
    video_path: str | Path,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    vid_stride: int = 1,
    save_video: bool = False,
    output_path: str | Path | None = None,
) -> dict:
    """
    Run inference on a video file, streaming frame-by-frame.

    Args:
        vid_stride: process every Nth frame (1 = every frame).
        save_video: if True, write an annotated output video.
        output_path: path for the annotated video (required when save_video=True).

    Returns:
        dict with video_path, total_frames, total_detections, elapsed_s,
        fps, and a per-frame results list.
    """
    video_path = Path(video_path)

    writer: cv2.VideoWriter | None = None
    if save_video:
        if output_path is None:
            output_path = video_path.with_stem(video_path.stem + "_annotated")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps / vid_stride, (w, h))

    per_frame: list[dict] = []
    total_detections = 0
    frame_idx = 0
    t0 = time.perf_counter()

    for result in model.predict(
        source=str(video_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        vid_stride=vid_stride,
        save=False,
        stream=True,
        verbose=False,
    ):
        dets = _extract_detections(result)
        total_detections += len(dets)
        per_frame.append({
            "frame": frame_idx,
            "detections": dets,
        })

        if writer is not None:
            writer.write(result.plot())

        frame_idx += 1

    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - t0
    return {
        "video_path": str(video_path),
        "total_frames": frame_idx,
        "total_detections": total_detections,
        "elapsed_s": round(elapsed, 2),
        "fps": round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0,
        "frames": per_frame,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_annotated(image: np.ndarray, output_path: str | Path) -> None:
    """Save an annotated BGR image to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated image to {output_path}")


def print_image_summary(result: dict) -> None:
    """Print a human-readable summary for a single-image result."""
    dets = result["detections"]
    h, w = result["image_shape"][:2]
    print(f"Image: {result['image_path']}  ({w}x{h})")
    print(f"Detections: {len(dets)}")
    for i, d in enumerate(dets):
        print(
            f"  {i+1}. {d['class_name']:>15s}  "
            f"conf={d['confidence']:.3f}  "
            f"xyxy={d['bbox_xyxy']}"
        )


def print_batch_summary(result: dict) -> None:
    """Print a human-readable summary for a batch/directory result."""
    print(f"Directory: {result['directory']}")
    print(
        f"  {result['total_images']} images, "
        f"{result['total_detections']} detections, "
        f"{result['elapsed_s']}s ({result['fps']} img/s)"
    )
    for img in result["results"]:
        print(f"  {img['file']}: {len(img['detections'])} detection(s)")


def print_video_summary(result: dict) -> None:
    """Print a human-readable summary for a video result."""
    print(f"Video: {result['video_path']}")
    print(
        f"  {result['total_frames']} frames, "
        f"{result['total_detections']} detections, "
        f"{result['elapsed_s']}s ({result['fps']} FPS)"
    )


# ---------------------------------------------------------------------------
# Main — demo usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Configuration (edit these) ---- #
    WEIGHTS = "path/to/your/best.pt"
    CONF_THRESHOLD = 0.25
    IMG_SIZE = 640
    # ------------------------------------ #

    model = load_model(WEIGHTS)

    # --- Single image ---
    # img_result = predict_image(model, "path/to/image.jpg", conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
    # print_image_summary(img_result)
    # save_annotated(img_result["annotated_image"], "runs/predict/result.jpg")

    # --- Directory of images ---
    # batch_result = predict_directory(model, "path/to/images/", conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
    # print_batch_summary(batch_result)
    # for img in batch_result["results"]:
    #     save_annotated(img["annotated_image"], f"runs/predict/{img['file']}")

    # --- Video ---
    # video_result = predict_video(
    #     model, "path/to/video.mp4",
    #     conf=CONF_THRESHOLD, imgsz=IMG_SIZE,
    #     vid_stride=2,
    #     save_video=True, output_path="runs/predict/output.mp4",
    # )
    # print_video_summary(video_result)

    print("Edit the __main__ block to uncomment the mode you want to run.")
