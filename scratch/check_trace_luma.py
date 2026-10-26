import json
import cv2
import numpy as np

with open("experiments/agentic_traces/golden_10_verified_traces.json", "r") as f:
    traces = json.load(f)

for i, t in enumerate(traces, 1):
    img_id = t["image_id"]
    full_path = f"artifacts/golden_traces/trace_{i:02d}_full.jpg"
    img = cv2.imread(full_path)
    if img is None:
        print(f"Trace {i:02d}: {full_path} NOT FOUND")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    print(f"Trace {i:02d} ({img_id}): mean_luma={mean_luma:.1f}, tag={t.get('environment_tag')}, branch={t.get('branch_type')}, tool={t.get('mcp_tool_selected')}, scenario={t.get('scenario_title')}")
