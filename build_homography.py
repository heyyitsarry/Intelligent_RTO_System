import cv2
import json
import numpy as np

IMG_PATH = "reference_frame.jpg"
POINTS_FILE = "ground_points.json"
OUT_FILE = "homography.npy"

img = cv2.imread(IMG_PATH)
h, w = img.shape[:2]

with open(POINTS_FILE) as f:
    src_pts = np.array(json.load(f), dtype=np.float32)

# destination rectangle (top-down)
dst_pts = np.array([
    [0, 0],
    [800, 0],
    [800, 800],
    [0, 800]
], dtype=np.float32)

H, _ = cv2.findHomography(src_pts, dst_pts)
np.save(OUT_FILE, H)

print("✅ Homography saved as homography.npy")

# Preview
topdown = cv2.warpPerspective(img, H, (800, 800))
cv2.imshow("Top-Down View", topdown)
cv2.waitKey(0)
cv2.destroyAllWindows()
