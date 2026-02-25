import cv2
import json

IMAGE_PATH = "reference_frame.jpeg"
OUTPUT_JSON = "lane_polygons.json"

outer_lane = []
inner_lanes = []
current_inner = []

current_mode = "outer"

def mouse_callback(event, x, y, flags, param):
    global outer_lane, current_inner, current_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_mode == "outer":
            outer_lane.append((x, y))
            print("Outer:", (x, y))
        elif current_mode == "inner":
            current_inner.append((x, y))
            print("Inner:", (x, y))

def draw_poly(img, points, color):
    for i in range(len(points)):
        cv2.circle(img, points[i], 4, color, -1)
        if i > 0:
            cv2.line(img, points[i-1], points[i], color, 2)

img = cv2.imread(IMAGE_PATH)
cv2.namedWindow("Lane Calibration")
cv2.setMouseCallback("Lane Calibration", mouse_callback)

print("""
INSTRUCTIONS:
- Click points to draw polygons
- 'o' : outer boundary (RED)
- 'i' : inner boundary (BLUE)
- 'n' : finish current inner loop
- 's' : save
- 'q' : quit
""")

while True:
    display = img.copy()

    draw_poly(display, outer_lane, (0,0,255))

    for poly in inner_lanes:
        draw_poly(display, poly, (255,0,0))

    draw_poly(display, current_inner, (255,0,0))

    cv2.imshow("Lane Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('o'):
        current_mode = "outer"
        print("Mode: OUTER")

    elif key == ord('i'):
        current_mode = "inner"
        print("Mode: INNER")

    elif key == ord('n'):
        if len(current_inner) > 2:
            inner_lanes.append(current_inner.copy())
            print("Inner loop saved")
        current_inner = []

    elif key == ord('s'):
        if len(current_inner) > 2:
            inner_lanes.append(current_inner.copy())

        data = {
            "outer_lane": outer_lane,
            "inner_lanes": inner_lanes
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(data, f, indent=4)

        print("Saved lane polygons")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
