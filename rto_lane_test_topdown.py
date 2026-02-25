import cv2
import numpy as np
import json

# ================= CONFIG =================
VIDEO_PATH = "test_video.mp4"
GROUND_POINTS_FILE = "ground_points.json"
LANE_MASK_TOPDOWN_PATH = "lane_mask_topdown.png"

OUTPUT_SIZE = (900, 900)

CONFIRM_FRAMES = 5
PENALTY = 5
INITIAL_SCORE = 100

MIN_VEHICLE_AREA = 2500
BG_HISTORY = 500
BG_VAR_THRESHOLD = 40
# =========================================

# -------- Load ground points --------
with open(GROUND_POINTS_FILE, "r") as f:
    src_pts = np.array(json.load(f), dtype=np.float32)

dst_pts = np.array([
    [0, 0],
    [OUTPUT_SIZE[0], 0],
    [OUTPUT_SIZE[0], OUTPUT_SIZE[1]],
    [0, OUTPUT_SIZE[1]]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# -------- Load lane mask --------
lane_mask = cv2.imread(LANE_MASK_TOPDOWN_PATH, cv2.IMREAD_GRAYSCALE)
lane_mask = cv2.resize(lane_mask, OUTPUT_SIZE)
_, lane_mask = cv2.threshold(lane_mask, 127, 255, cv2.THRESH_BINARY)

# -------- Video --------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

bg = cv2.createBackgroundSubtractorMOG2(
    history=BG_HISTORY,
    varThreshold=BG_VAR_THRESHOLD,
    detectShadows=False
)

kernel = np.ones((5, 5), np.uint8)

# -------- State --------
score = INITIAL_SCORE
violation_streak = 0
inside_lane = True
started = False

print("▶️ Press 'S' to start test")

# =========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not started:
        started = True
        print("✅ TEST STARTED")

    if not started:
        cv2.putText(display, "PRESS 'S' TO START",
                    (250, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 255, 255), 3)
        cv2.imshow("RTO Driving Test", display)
        continue

    # -------- Foreground detection --------
    fg = bg.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(
        fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_contour = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_VEHICLE_AREA and area > max_area:
            vehicle_contour = c
            max_area = area

    violation = False

    if vehicle_contour is not None:
        M = cv2.moments(vehicle_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(display, (cx, cy), 6, (255, 0, 0), -1)

            pt = np.array([[[cx, cy]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(pt, H)[0][0]
            tx, ty = int(warped_pt[0]), int(warped_pt[1])

            if 0 <= tx < OUTPUT_SIZE[0] and 0 <= ty < OUTPUT_SIZE[1]:
                if lane_mask[ty, tx] == 0:
                    violation = True

            color = (0, 0, 255) if violation else (0, 255, 0)
            cv2.drawContours(display, [vehicle_contour], -1, color, 2)

    # -------- Scoring logic --------
    if violation:
        violation_streak += 1
    else:
        violation_streak = 0
        inside_lane = True

    if inside_lane and violation_streak >= CONFIRM_FRAMES:
        score -= PENALTY
        inside_lane = False
        violation_streak = 0
        print(f"❌ Lane violation | Score = {score}")

    if not inside_lane and not violation:
        inside_lane = True
        print("✅ Vehicle fully back inside lane")

    # ==================================================
    # 🎮 DIGITAL SCOREBOARD (NEW HUD)
    # ==================================================
    box_x, box_y = 20, 20
    box_w, box_h = 240, 110

    bg_color = (20, 20, 20)
    good_color = (0, 255, 0)
    bad_color = (0, 0, 255)

    score_color = good_color if score >= 60 else bad_color

    cv2.rectangle(display,
                  (box_x, box_y),
                  (box_x + box_w, box_y + box_h),
                  bg_color, -1)

    cv2.rectangle(display,
                  (box_x, box_y),
                  (box_x + box_w, box_y + box_h),
                  score_color, 3)

    cv2.putText(display, "SCORE",
                (box_x + 60, box_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (200, 200, 200), 2)

    cv2.putText(display, f"{score}",
                (box_x + 55, box_y + 90),
                cv2.FONT_HERSHEY_DUPLEX,
                2.8, score_color, 5)

    # ==================================================

    cv2.imshow("RTO Driving Test", display)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Test completed")
