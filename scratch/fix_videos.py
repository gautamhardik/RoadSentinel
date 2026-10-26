import subprocess
import os
import shutil
import json

videos = [
    "night_urban_vru.mp4",
    "rain_intersection.mp4",
    "distant_corridor_pedestrian.mp4",
    "heavy_occlusion_escalation.mp4"
]

artifacts_dir = "artifacts/demo_videos"

for v in videos:
    src_v = os.path.join(artifacts_dir, v)
    temp_v = os.path.join(artifacts_dir, f"temp_{v}")
    
    cmd = [
        "ffmpeg", "-y", "-i", src_v,
        "-vf", "scale=in_range=full:out_range=tv,format=yuv420p",
        "-color_range", "tv",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-movflags", "+faststart",
        temp_v
    ]
    print(f"Transcoding {v} to standard yuv420p...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        os.replace(temp_v, src_v)
        print(f"  [SUCCESS] {v} updated to standard yuv420p!")
    else:
        print(f"  [ERROR] Failed to transcode {v}: {res.stderr[:200]}")
        if os.path.exists(temp_v):
            os.remove(temp_v)

# Verify with ffprobe
print("\n--- FFPROBE VERIFICATION ---")
all_ok = True
for v in videos:
    src_v = os.path.join(artifacts_dir, v)
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height",
        "-of", "json", src_v
    ]
    res = subprocess.run(probe_cmd, capture_output=True, text=True)
    st = json.loads(res.stdout)["streams"][0]
    codec = st.get("codec_name")
    pix_fmt = st.get("pix_fmt")
    w = st.get("width")
    h = st.get("height")
    print(f"  {v}: codec={codec}, pix_fmt={pix_fmt}, resolution={w}x{h}")
    assert codec == "h264", f"Codec {codec} != h264"
    assert pix_fmt == "yuv420p", f"pix_fmt {pix_fmt} != yuv420p"
    assert w == 1920 and h == 648, f"resolution {w}x{h} != 1920x648"

print("\n>>> ALL 4 VIDEOS VERIFIED 100% STANDARD YUV420P H.264 1920x648 <<<")
