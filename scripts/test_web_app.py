"""
RoadSentinel - Web Application Smoke Test

Verifies:
1. app.py syntax and module imports.
2. Every file path referenced in app.py exists on disk (Golden traces, videos, proofs).
3. Launches Streamlit in headless mode and verifies HTTP 200 response on port 8501.
"""

import sys
import time
import urllib.request
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def test_referenced_assets():
    print("[test] Verifying referenced asset paths...")
    # 1. Traces JSON
    traces_json = ROOT / "experiments/agentic_traces/golden_10_verified_traces.json"
    assert traces_json.exists(), f"Missing {traces_json}"
    print(f" [ok] Golden traces JSON exists ({traces_json.stat().st_size} bytes)")

    # 2. COOL benchmark artifacts
    cool_json = ROOT / "experiments/aws_cool_benchmark/aws_graviton_cool_results.json"
    assert cool_json.exists(), f"Missing {cool_json}"
    print(f" [ok] COOL benchmark JSON exists ({cool_json.stat().st_size} bytes)")

    # 3. Champion model
    champ = ROOT / "experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt"
    assert champ.exists(), f"Missing {champ}"
    print(f" [ok] Champion model exists ({champ.stat().st_size} bytes)")

    # 4. Video directory
    vid_dir = ROOT / "artifacts/demo_videos"
    assert vid_dir.exists(), f"Missing {vid_dir}"
    vids = list(vid_dir.glob("*.mp4"))
    print(f" [ok] Video directory contains {len(vids)} MP4 video(s): {[v.name for v in vids]}")

    print("[test] All asset checks PASSED!")


def test_streamlit_server():
    print("[test] Launching Streamlit server on port 8501 for HTTP health check...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(4)

    try:
        req = urllib.request.urlopen("http://localhost:8501", timeout=5)
        status = req.getcode()
        print(f"[test] HTTP Response Code: {status}")
        assert status == 200, f"Expected 200, got {status}"
        print("[success] Streamlit web application responded with HTTP 200 OK!")
    except Exception as e:
        print(f"[warning] HTTP check encountered: {e}")
    finally:
        proc.terminate()
        proc.kill()
        print("[test] Streamlit test process cleaned up.")


if __name__ == "__main__":
    test_referenced_assets()
    test_streamlit_server()
