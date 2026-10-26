"""Quick-start: load YOLO26 with custom weights and run on a single image."""

from ultralytics import YOLO

WEIGHTS = "path/to/your/best.pt"   # <-- replace with your custom weights
IMAGE = "path/to/your/image.jpg"   # <-- replace with your test image


def main() -> None:
    model = YOLO(WEIGHTS)

    results = model.predict(
        source=IMAGE,
        imgsz=640,
        conf=0.25,
        save=True,
        project="runs",
        name="predict",
    )

    for det in results[0].boxes:
        cls_id = int(det.cls.item())
        label = results[0].names[cls_id]
        conf = det.conf.item()
        bbox = det.xyxy[0].tolist()
        print(f"{label}: {conf:.2f}  bbox={[round(v, 1) for v in bbox]}")


if __name__ == "__main__":
    main()
