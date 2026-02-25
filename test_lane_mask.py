import cv2
import json
import numpy as np

# ---------------- CONFIG ----------------
IMAGE_PATH = "reference_frame.jpeg"
LANE_JSON = "lane_polygons.json"
MARGIN_PX = 15   # safety margin (tune later)

# ---------------------------------------
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

with open(LANE_JSON, "r") as f:
    data = json.load(f)

outer_lane = np.array(data["outer_lane"], dtype=np.int32)

inner_lanes = [
    np.array(poly, dtype=np.int32)
    for poly in data["inner_lanes"]
]

# 1. Create empty mask
lane_mask = np.zeros((h, w), dtype=np.uint8)

# 2. Fill outer lane (white)
cv2.fillPoly(lane_mask, [outer_lane], 255)

# 3. Remove inner lanes (black holes)
for inner in inner_lanes:
    cv2.fillPoly(lane_mask, [inner], 0)

# 4. Shrink lane for safety margin
kernel = np.ones((MARGIN_PX, MARGIN_PX), np.uint8)
safe_lane_mask = cv2.erode(lane_mask, kernel)
cv2.imwrite("safe_lane_mask.png", safe_lane_mask)
print("safe_lane_mask.png saved")

# 5. Visualize
overlay = img.copy()
overlay[safe_lane_mask == 0] = (0, 0, 0)

combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

cv2.imshow("Original", img)
cv2.imshow("Valid Lane Mask", lane_mask)
cv2.imshow("Safe Lane (Margin Applied)", safe_lane_mask)
cv2.imshow("Overlay", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
