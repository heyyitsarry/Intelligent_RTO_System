import cv2
import numpy as np

SIZE = 800
mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

# DRAW ALLOWED AREA (white)
# Use cv2.fillPoly / cv2.circle / cv2.rectangle manually
# Example (you will adjust):

# Left loop
cv2.circle(mask, (250, 400), 180, 255, -1)
# Right loop
cv2.circle(mask, (550, 400), 180, 255, -1)
# Middle connector
cv2.rectangle(mask, (250, 220), (550, 580), 255, -1)

cv2.imshow("Lane Mask", mask)
cv2.imwrite("lane_mask_topdown.png", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
