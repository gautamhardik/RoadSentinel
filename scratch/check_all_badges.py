import cv2
import numpy as np

for i in range(1, 13):
    raw_path = f"artifacts/golden_traces/trace_{i:02d}_raw_crop.jpg"
    cv_path = f"artifacts/golden_traces/trace_{i:02d}_opencv_crop.jpg"
    raw_img = cv2.imread(raw_path)
    cv_img = cv2.imread(cv_path)
    
    # Check if banner rectangle or text is in top rows
    # In generate_verified_agent_traces.py, badge rectangle was filled with (11, 15, 23)
    has_raw_badge = (raw_img[:50, :250] == [11, 15, 23]).all(axis=2).sum() > 200
    has_cv_badge = (cv_img[:50, :250] == [11, 15, 23]).all(axis=2).sum() > 200
    print(f"Trace {i:02d}: raw_badge={has_raw_badge}, cv_badge={has_cv_badge}")
