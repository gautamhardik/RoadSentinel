import cv2
import numpy as np

img = cv2.imread("artifacts/golden_traces/trace_01_raw_crop.jpg")
print("trace_01_raw_crop.jpg shape:", img.shape)
# Inspect top 40 rows
top = img[:40, :]
print("Top 40 rows min/max/mean:", top.min(), top.max(), top.mean())

# Check generate_verified_agent_traces.py to see how crops were created
with open("scripts/generate_verified_agent_traces.py", "r") as f:
    code = f.read()
    if "cv2.putText" in code:
        print("Found cv2.putText in generate_verified_agent_traces.py!")
        for line in code.splitlines():
            if "putText" in line:
                print("  ", line.strip())
