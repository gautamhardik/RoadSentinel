import json

with open("experiments/agentic_traces/golden_10_verified_traces.json", "r") as f:
    traces = json.load(f)

for i, t in enumerate(traces, 1):
    print(f"--- Trace {i:02d} keys: {list(t.keys())}")
    for k in ['image_id', 'scenario_title', 'condition', 'decision', 'action', 'initial_conf', 'verified_conf', 'conf_raw', 'conf_opencv', 'crop_image_url', 'enhanced_crop_image_url', 'full_image_url', 'bbox_xyxy']:
        if k in t:
            print(f"    {k}: {t[k]}")
