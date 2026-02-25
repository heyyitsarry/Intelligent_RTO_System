import cv2
import numpy as np

topdown = cv2.imread("topdown_reference.png")
mask = cv2.imread("lane_mask_topdown.png", 0)

overlay = topdown.copy()
overlay[mask == 255] = (0, 255, 0)

cv2.imshow(
    "Lane Overlay Check",
    cv2.addWeighted(topdown, 0.7, overlay, 0.3, 0)
)
cv2.waitKey(0)
cv2.destroyAllWindows()
