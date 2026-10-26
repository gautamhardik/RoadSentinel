import os
import json

with open("experiments/agentic_traces/golden_10_verified_traces.json") as f:
    traces = json.load(f)

print(f"Total traces: {len(traces)}")
all_ok = True
for t in traces:
    tid = t["trace_id"]
    full = t["full_image_url"].lstrip("/")
    raw = t["raw_crop_url"].lstrip("/")
    cv_img = t["opencv_crop_url"].lstrip("/")
    
    f_ok = os.path.exists(full) and os.path.getsize(full) > 5000
    r_ok = os.path.exists(raw) and os.path.getsize(raw) > 5000
    c_ok = os.path.exists(cv_img) and os.path.getsize(cv_img) > 5000
    
    if not (f_ok and r_ok and c_ok):
        all_ok = False
        print(f"FAIL Trace {tid:02d}: full={f_ok} ({full}), raw={r_ok} ({raw}), opencv={c_ok} ({cv_img})")
    else:
        print(f"OK Trace {tid:02d}: {t['image_id']} (delta={t['counterfactual_analysis']['opencv_delta']:+.3f})")

print(f"ALL TRACE IMAGES VERIFIED: {all_ok}")
