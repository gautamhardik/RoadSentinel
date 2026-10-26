# yolo-bdd100k

YOLO26 object detection inference toolkit for the BDD100K dataset. Load custom-trained weights and run predictions on single images, directories of images, or video files.

## Requirements

- Python >= 3.10
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO26)
- OpenCV

## Setup

```bash
# Clone the repo
git clone <repo-url> && cd yolo-bdd100k

# Install dependencies (pick one)
pip install -e .
# or with uv
uv sync
```

## Project Structure

```
yolo-bdd100k/
├── main.py               # Minimal quick-start script
├── predict.py             # Full CLI tool (image / directory / video)
├── predict_detailed.py    # Importable library with reusable functions
├── pyproject.toml
└── README.md
```

## Quick Start

Edit the constants at the top of `main.py` and run:

```bash
python main.py
```

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

results = model.predict(source="photo.jpg", imgsz=640, conf=0.25, save=True)

for det in results[0].boxes:
    cls_id = int(det.cls.item())
    label = results[0].names[cls_id]
    conf = det.conf.item()
    bbox = det.xyxy[0].tolist()
    print(f"{label}: {conf:.2f}  bbox={[round(v, 1) for v in bbox]}")
```

## CLI Usage — `predict.py`

`predict.py` auto-detects whether `--source` is a single image, a directory, or a video file, and runs the appropriate inference mode.

### Single Image

```bash
python predict.py \
  --weights runs/train/best.pt \
  --source test.jpg
```

### Batch — Directory of Images

```bash
python predict.py \
  --weights runs/train/best.pt \
  --source ./test_images/ \
  --batch-size 32 \
  --save-json
```

### Video

```bash
python predict.py \
  --weights runs/train/best.pt \
  --source dashcam.mp4 \
  --vid-stride 2 \
  --conf 0.4
```

### All CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--weights` | *(required)* | Path to custom `.pt` weights |
| `--source` | *(required)* | Image path, image directory, or video file |
| `--img-size` | `640` | Inference resolution in pixels |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.7` | IoU threshold (one-to-many head) |
| `--device` | auto | `cpu`, `0` (GPU), or `mps` |
| `--save-dir` | `runs/predict` | Output directory for annotated results |
| `--save-txt` | off | Save YOLO-format `.txt` label files |
| `--save-json` | off | Write `results.json` with all detections |
| `--show` | off | Display annotated result on screen |
| `--no-save` | off | Skip saving annotated images/video |
| `--no-end2end` | off | Use one-to-many head (with NMS) instead of the default NMS-free head |
| `--batch-size` | `16` | Batch size for directory mode |
| `--vid-stride` | `1` | Process every Nth video frame |

### Supported File Types

| Type | Extensions |
|---|---|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp` |
| Videos | `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v` |

## Library Usage — `predict_detailed.py`

Import the functions directly for programmatic use in notebooks, pipelines, or other scripts.

### Single Image

```python
from predict_detailed import load_model, predict_image, print_image_summary, save_annotated

model = load_model("best.pt", device="mps")

result = predict_image(model, "photo.jpg", conf=0.3, imgsz=640)
print_image_summary(result)
save_annotated(result["annotated_image"], "output/result.jpg")
```

### Batch — Directory of Images

```python
from predict_detailed import load_model, predict_directory, print_batch_summary, save_annotated

model = load_model("best.pt")

batch = predict_directory(model, "./test_images/", conf=0.25, batch_size=32)
print_batch_summary(batch)

for img in batch["results"]:
    save_annotated(img["annotated_image"], f"output/{img['file']}")
```

### Video

```python
from predict_detailed import load_model, predict_video, print_video_summary

model = load_model("best.pt")

result = predict_video(
    model,
    "dashcam.mp4",
    conf=0.3,
    vid_stride=2,
    save_video=True,
    output_path="output/annotated.mp4",
)
print_video_summary(result)
```

### Return Value Reference

All functions return plain dicts that can be serialized to JSON.

**`predict_image()` returns:**

```python
{
    "image_path": "photo.jpg",
    "image_shape": (720, 1280),
    "detections": [
        {
            "class_id": 0,
            "class_name": "car",
            "confidence": 0.9312,
            "bbox_xyxy": [100.5, 200.0, 400.3, 550.8],
            "bbox_xywh": [250.4, 375.4, 299.8, 350.8],
        },
    ],
    "annotated_image": np.ndarray,  # BGR, ready for cv2.imwrite
}
```

**`predict_directory()` returns:**

```python
{
    "directory": "./test_images",
    "total_images": 50,
    "total_detections": 237,
    "elapsed_s": 4.21,
    "fps": 11.9,
    "results": [ ... ],  # list of per-image dicts (same shape as predict_image)
}
```

**`predict_video()` returns:**

```python
{
    "video_path": "dashcam.mp4",
    "total_frames": 900,
    "total_detections": 4520,
    "elapsed_s": 30.5,
    "fps": 29.5,
    "frames": [
        {"frame": 0, "detections": [ ... ]},
        {"frame": 1, "detections": [ ... ]},
    ],
}
```

## YOLO26 Model Weights

I have trained the YOLO26 Small model on the entire BDD100k Training dataset with old nomenclature. You can get the weights from huggingface repository:


[shravanda/yolo26-bdd100k](https://huggingface.co/shravanda/yolo26-bdd100k/tree/main)

### Other variants

| Variant | Weights File | Use Case |
|---|---|---|
| Nano | `yolo26n.pt` | Edge / mobile, fastest |
| Small | `yolo26s.pt` | Balanced speed & accuracy |
| Medium | `yolo26m.pt` | Higher accuracy |
| Large | `yolo26l.pt` | High accuracy |
| XLarge | `yolo26x.pt` | Maximum accuracy |

YOLO26 is natively NMS-free (end-to-end). Pass `--no-end2end` to use the one-to-many head with NMS if you need slightly higher mAP at the cost of extra post-processing.

## License

See [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (AGPL-3.0).
