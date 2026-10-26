import cv2
import numpy as np

raw = cv2.imread("artifacts/golden_traces/trace_01_raw_crop.jpg")
clean = cv2.imread("artifacts/golden_traces/trace_01_raw.jpg")
print("raw shape:", raw.shape, "clean shape:", clean.shape if clean is not None else None)

# Check if there is a rectangle or lines drawn in raw
# In draw_targeting_box: color_bgr was (11, 158, 245) or (129, 185, 16)
# Let's check for near-amber pixels: B ~ 11, G ~ 158, R ~ 245
amber = (np.abs(raw[:, :, 0].astype(int) - 11) < 15) & (np.abs(raw[:, :, 1].astype(int) - 158) < 25) & (np.abs(raw[:, :, 2].astype(int) - 245) < 25)
print("Amber line pixels in trace_01_raw_crop.jpg:", amber.sum())

cv_crop = cv2.imread("artifacts/golden_traces/trace_01_opencv_crop.jpg")
emerald = (np.abs(cv_crop[:, :, 0].astype(int) - 129) < 25) & (np.abs(cv_crop[:, :, 1].astype(int) - 185) < 25) & (np.abs(cv_crop[:, :, 2].astype(int) - 16) < 20)
print("Emerald line pixels in trace_01_opencv_crop.jpg:", emerald.sum())
