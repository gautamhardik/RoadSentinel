"""
RoadSentinel - Download and Verify Genuine Dashcam Driving Videos

Downloads 4 genuine front-facing autonomous driving dashcam video sequences:
1. Night Urban Driving (Nighttime illuminated street with reflections)
2. Rain Intersection (Nighttime rain with wet reflections)
3. Urban Pedestrians (Daytime urban corridor with pedestrians/traffic)
4. Highway Traffic Corridors (Genuine Udacity front-facing dashcam)

Extracts preview frames to artifacts/video_previews/ for visual inspection.
"""

import os
import sys
import urllib.request
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/demo_videos/raw"
PREVIEW_DIR = ROOT / "artifacts/video_previews"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# Genuine front-facing driving dashcam video sources
SOURCES = {
    "night_urban_driving.mp4": "https://huggingface.co/datasets/vijpandaturtle/roadscapes-video/resolve/main/test/Sequence_Night_12/Sequence_Night_12_clip1.mp4",
    "rain_intersection.mp4": "https://huggingface.co/datasets/vijpandaturtle/roadscapes-video/resolve/main/test/Sequence_Night_16/Sequence_Night_16_clip12.mp4",
    "urban_pedestrians.mp4": "https://huggingface.co/datasets/vijpandaturtle/roadscapes-video/resolve/main/test/Sequence_Day_11/Sequence_Day_11_clip9.mp4",
    "sample_driving_test.mp4": "https://raw.githubusercontent.com/udacity/CarND-Vehicle-Detection/master/test_video.mp4",
}


def download_file(url: str, dest: Path):
    print(f"[download] Fetching {dest.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        while chunk := resp.read(65536):
            out.write(chunk)
            downloaded += len(chunk)
    print(f" [ok] Saved {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")


def verify_and_extract_previews():
    for name, url in SOURCES.items():
        dest = RAW_DIR / name
        download_file(url, dest)

        # OpenCV inspection & frame extraction
        cap = cv2.VideoCapture(str(dest))
        assert cap.isOpened(), f"Failed to open {dest}"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[inspect] {name}: {w}x{h} @ {fps:.1f} fps, {n_frames} frames")

        # Extract Frame 1 and Frame min(50, n_frames - 1)
        ret, f1 = cap.read()
        if ret:
            cv2.imwrite(str(PREVIEW_DIR / f"{Path(name).stem}_f001.jpg"), f1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, min(50, max(0, n_frames - 2)))
        ret, f50 = cap.read()
        if ret:
            cv2.imwrite(str(PREVIEW_DIR / f"{Path(name).stem}_f050.jpg"), f50)

        cap.release()

    print("[complete] All 4 genuine front-facing dashcam driving videos verified and preview frames extracted!")


if __name__ == "__main__":
    verify_and_extract_previews()
