import cv2
import numpy as np

VIDEO_PATH = "test_video.mp4"
HOMOGRAPHY_PATH = "homography.npy"
TOPDOWN_SIZE = 800

cap = cv2.VideoCapture(VIDEO_PATH)
H = np.load(HOMOGRAPHY_PATH)

# Read first valid frame
while True:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ No frame found in video")
    if frame is not None:
        break

topdown = cv2.warpPerspective(frame, H, (TOPDOWN_SIZE, TOPDOWN_SIZE))

cv2.imwrite("topdown_reference.png", topdown)
cv2.imshow("Top-Down Reference", topdown)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

print("✅ Saved topdown_reference.png")
