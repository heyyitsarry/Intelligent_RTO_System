import cv2
import json

IMG_PATH = "reference_frame.jpg"
POINTS_FILE = "ground_points.json"

points = []

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point {len(points)}: {x}, {y}")

img = cv2.imread(IMG_PATH)
clone = img.copy()

cv2.namedWindow("Click 4 ground points")
cv2.setMouseCallback("Click 4 ground points", mouse_cb)

while True:
    display = clone.copy()
    for p in points:
        cv2.circle(display, tuple(p), 6, (0, 0, 255), -1)

    cv2.imshow("Click 4 ground points", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') and len(points) == 4:
        break

cv2.destroyAllWindows()

with open(POINTS_FILE, "w") as f:
    json.dump(points, f)

print("✅ Saved ground points to", POINTS_FILE)
