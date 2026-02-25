import cv2
import numpy as np

# ================= CONFIG =================
VIDEO_PATH = "test_video.mp4"
SAFE_LANE_MASK_PATH = "safe_lane_mask.png"

# Background subtraction
BG_HISTORY = 500
BG_VAR_THRESHOLD = 50
WARMUP_FRAMES = 40

# Vehicle detection
MIN_VEHICLE_AREA = 2000

# Dense-core extraction (shadow suppression)
CORE_DENSITY_THRESHOLD = 0.5   # tune 0.45–0.55

# Violation logic
OVERLAP_PIXEL_THRESHOLD = 25
OUTSIDE_CONFIRM_FRAMES = 3     # confirm exit
INSIDE_CONFIRM_FRAMES = 8      # confirm re-entry

# Scoring
PENALTY = 5
INITIAL_SCORE = 100

# =========================================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

lane_mask = cv2.imread(SAFE_LANE_MASK_PATH, cv2.IMREAD_GRAYSCALE)
if lane_mask is None:
    raise RuntimeError("❌ Cannot load lane mask")

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=BG_HISTORY,
    varThreshold=BG_VAR_THRESHOLD,
    detectShadows=True
)

kernel = np.ones((5, 5), np.uint8)

# -------- STATE --------
frame_idx = 0
score = INITIAL_SCORE

test_started = False
test_active = False   # vehicle fully inside lane

outside_latched = False
outside_confirm = 0
inside_confirm = 0

print("▶️ Press 'S' to start test")

# =========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]
    display = frame.copy()

    lane = cv2.resize(lane_mask, (w, h))
    forbidden = cv2.bitwise_not(lane)

    # -------- Foreground extraction --------
    fg = bg_subtractor.apply(frame)
    fg[fg == 127] = 0  # remove shadows
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel)

    # -------- Warmup --------
    if frame_idx < WARMUP_FRAMES:
        cv2.putText(display, "INITIALIZING...",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 2)
        cv2.imshow("RTO Driving Test", display)
        cv2.waitKey(30)
        continue

    # -------- Detect largest moving object --------
    contours, _ = cv2.findContours(
        fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_VEHICLE_AREA and area > max_area:
            max_area = area
            vehicle_contour = cnt

    violation_now = False
    core_mask = None

    if vehicle_contour is not None:
        # ---- vehicle mask ----
        vehicle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(vehicle_mask, [vehicle_contour], -1, 255, -1)

        # ---- dense core ----
        dist = cv2.distanceTransform(vehicle_mask, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        core_mask = np.zeros_like(vehicle_mask)
        core_mask[dist_norm > CORE_DENSITY_THRESHOLD] = 255

        core_overlap = cv2.bitwise_and(core_mask, forbidden)
        overlap_pixels = cv2.countNonZero(core_overlap)

        # activate test only when fully inside
        if test_started and not test_active and overlap_pixels == 0:
            test_active = True
            print("✅ Vehicle entered test lane")

        if test_active and overlap_pixels > OVERLAP_PIXEL_THRESHOLD:
            violation_now = True

        # draw contour
        color = (0, 0, 255) if violation_now else (0, 255, 0)
        cv2.drawContours(display, [vehicle_contour], -1, color, 2)

        # visualize core safely (blue)
        blue_overlay = np.zeros_like(display)
        blue_overlay[:, :, 0] = core_mask
        display = cv2.addWeighted(display, 1.0, blue_overlay, 0.4, 0)

    # -------- STATE MACHINE (FIXED) --------
    if test_started and test_active:
        if violation_now:
            outside_confirm += 1
            inside_confirm = 0
        else:
            inside_confirm += 1
            outside_confirm = 0

        # CONFIRM EXIT (one-time penalty)
        if not outside_latched and outside_confirm >= OUTSIDE_CONFIRM_FRAMES:
            score -= PENALTY
            outside_latched = True
            outside_confirm = 0
            print(f"❌ Lane violation | Score = {score}")

        # CONFIRM RE-ENTRY (unlatch)
        if outside_latched and inside_confirm >= INSIDE_CONFIRM_FRAMES:
            outside_latched = False
            inside_confirm = 0
            print("✅ Vehicle fully back inside lane")

    # -------- UI --------
    cv2.putText(display, f"Score: {score}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2)

    if not test_started:
        cv2.putText(display, "PRESS 'S' TO START TEST",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

    cv2.imshow("RTO Driving Test", display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('s') and not test_started:
        test_started = True
        test_active = False
        outside_latched = False
        outside_confirm = 0
        inside_confirm = 0
        score = INITIAL_SCORE
        print("✅ TEST STARTED")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Test completed")
