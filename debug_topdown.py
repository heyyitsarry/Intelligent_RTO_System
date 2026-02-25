import cv2
import numpy as np

IMG_PATH = "reference_frame.jpg"
H = np.load("homography.npy")

img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("Cannot load image")

topdown = cv2.warpPerspective(img, H, (1000, 1000))

cv2.imshow("Original", img)
cv2.imshow("Top-Down View", topdown)

cv2.waitKey(0)
cv2.destroyAllWindows()
