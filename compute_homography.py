import json
import cv2
import numpy as np

# Load clicked points
with open("ground_points.json", "r") as f:
    src_pts = np.array(json.load(f), dtype=np.float32)

# Define real-world top-down rectangle
# (units are arbitrary but consistent)
dst_pts = np.array([
    [0, 0],
    [1000, 0],
    [1000, 1000],
    [0, 1000]
], dtype=np.float32)

# Compute homography
H, _ = cv2.findHomography(src_pts, dst_pts)

# Save matrix
np.save("homography.npy", H)

print("✅ Homography matrix saved as homography.npy")
print("\nH =\n", H)
