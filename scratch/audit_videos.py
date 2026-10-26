import cv2
import json
import subprocess
import os

videos = [
    "night_urban_vru.mp4",
    "rain_intersection.mp4",
    "distant_corridor_pedestrian.mp4",
    "heavy_occlusion_escalation.mp4"
]

print("=== PILLAR 3: VIDEO PIPELINE AUDIT ===")
for v in videos:
    vpath = os.path.join("artifacts/demo_videos", v)
    if not os.path.exists(vpath):
        print(f"ERROR: {vpath} does not exist!")
        continue

    # ffprobe check
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height",
        "-of", "json", vpath
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(res.stdout)["streams"][0]
        codec = probe.get("codec_name")
        pix_fmt = probe.get("pix_fmt")
        w = probe.get("width")
        h = probe.get("height")
        print(f"{v}: codec={codec}, pix_fmt={pix_fmt}, resolution={w}x{h}")
    except Exception as e:
        print(f"ffprobe failed for {v}: {e}")

    # cv2 check and frame 30 extraction
    cap = cv2.VideoCapture(vpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    print(f"  cv2 read: total_frames={total_frames}, fps={fps}, frame30={frame.shape if ret else None}")

# Check video_benchmark_summary.json
sum_path = "artifacts/demo_videos/video_benchmark_summary.json"
if os.path.exists(sum_path):
    with open(sum_path, 'r') as f:
        summary = json.load(f)
    print("video_benchmark_summary.json loaded:")
    for item in summary.get("videos", []):
        name = item.get("filename")
        base_vru = item.get("baseline_vru_detections")
        rs_vru = item.get("roadsentinel_vru_detections")
        tool = item.get("primary_opencv_tool")
        print(f"  {name}: baseline={base_vru}, roadsentinel={rs_vru}, tool={tool}")
else:
    print(f"ERROR: {sum_path} not found!")
