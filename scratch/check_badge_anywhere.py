import cv2
import numpy as np
import json

with open("experiments/agentic_traces/golden_10_verified_traces.json", "r") as f:
    traces = json.load(f)

for i in range(1, 13):
    raw_path = f"artifacts/golden_traces/trace_{i:02d}_raw_crop.jpg"
    cv_path = f"artifacts/golden_traces/trace_{i:02d}_opencv_crop.jpg"
    raw_img = cv2.imread(raw_path)
    cv_img = cv2.imread(cv_path)
    
    # Check if (11, 15, 23) or pure white text pixels exist anywhere in the image
    dark_badge_raw = ((raw_img[:, :, 0] == 23) & (raw_img[:, :, 1] == 15) & (raw_img[:, :, 2] == 11)).sum()
    dark_badge_cv = ((cv_img[:, :, 0] == 23) & (cv_img[:, :, 1] == 15) & (cv_img[:, :, 2] == 11)).sum()
    print(f"Trace {i:02d}: dark_badge_raw={dark_badge_raw}, dark_badge_cv={dark_badge_cv}")
