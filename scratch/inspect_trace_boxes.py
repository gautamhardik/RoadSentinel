import json

with open("experiments/agentic_traces/golden_10_verified_traces.json", "r") as f:
    traces = json.load(f)

for i, t in enumerate(traces, 1):
    print(f"Trace {i:02d}: {t['image_id']}, gt_box={t.get('gt_box')}, gt_class={t.get('gt_class')}, branch={t.get('branch_type')}")
