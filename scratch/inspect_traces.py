import json
import os
import cv2

with open("experiments/agentic_traces/golden_10_verified_traces.json", "r") as f:
    traces = json.load(f)

print("Trace count:", len(traces))
image_ids = [t["image_id"] for t in traces]
print("Unique image IDs:", len(set(image_ids)))

for i, t in enumerate(traces, 1):
    img_id = t["image_id"]
    scenario = t.get("scenario_title", "N/A")
    env = t.get("environmental_condition", "N/A")
    action = t.get("action_type", "N/A")
    raw_conf = t.get("confidence_raw", "N/A")
    cv_conf = t.get("confidence_opencv", "N/A")
    print(f"Trace {i:02d}: id={img_id}, scenario={scenario}, env={env}, action={action}, raw={raw_conf}, cv={cv_conf}")
