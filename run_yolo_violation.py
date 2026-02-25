import cv2
import numpy as np
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = "test_video.mp4"
SAFE_LANE_MASK_PATH = "safe_lane_mask.png"
MODEL_PATH = "yolov8n.pt"   # nano for CPU real-time

VEHICLE_CLASSES = {2, 3, 5, 7}
CONF_THRESH = 0.3

INITIAL_SCORE = 100
LANE_PENALTY = 5

# Performance
YOLO_FRAME_INTERVAL = 5

# Violation logic
VIOLATION_CONFIRM_FRAMES = 4
VIOLATION_RATIO_THRESH = 0.03     # 3% of car core outside
BBOX_SHRINK_RATIO = 0.2           # evaluate only car "core"

# =========================================

model = YOLO(MODEL_PATH)

safe_lane_mask = cv2.imread(SAFE_LANE_MASK_PATH, cv2.IMREAD_GRAYSCALE)

# ---- Add tolerance band (human realism) ----
kernel = np.ones((7, 7), np.uint8)
safe_lane_mask = cv2.dilate(safe_lane_mask, kernel, iterations=1)

h_mask, w_mask = safe_lane_mask.shape

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)

# -------- STATE --------
score = INITIAL_SCORE
inside_lane = False
test_started = False

last_bbox = None
frame_count = 0
violation_frame_count = 0

# =========================================
def check_violation(bbox):
    """
    Evaluate violation using shrunk car-core area + ratio
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    dx = int(w * BBOX_SHRINK_RATIO)
    dy = int(h * BBOX_SHRINK_RATIO)

    sx1 = max(0, x1 + dx)
    sy1 = max(0, y1 + dy)
    sx2 = min(w_mask, x2 - dx)
    sy2 = min(h_mask, y2 - dy)

    region = safe_lane_mask[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return False

    black_ratio = np.mean(region == 0)
    return black_ratio > VIOLATION_RATIO_THRESH

# =========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    best_bbox = None

    # ---------- YOLO DETECTION (SKIPPED FRAMES) ----------
    if frame_count % YOLO_FRAME_INTERVAL == 0:
        results = model(frame, imgsz=640, verbose=False)[0]
        max_area = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in VEHICLE_CLASSES and conf > CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    best_bbox = (x1, y1, x2, y2)

        if best_bbox is not None:
            last_bbox = best_bbox
    else:
        best_bbox = last_bbox

    # ---------- Disable bbox before manual start ----------
    if not test_started:
        best_bbox = None

    # ---------- VIOLATION LOGIC ----------
    if test_started and best_bbox is not None:
        violation_now = check_violation(best_bbox)

        if violation_now:
            violation_frame_count += 1
        else:
            violation_frame_count = 0
            inside_lane = True

        if inside_lane and violation_frame_count >= VIOLATION_CONFIRM_FRAMES:
            score -= LANE_PENALTY
            inside_lane = False
            violation_frame_count = 0
            print(f"Lane violation confirmed | Score = {score}")

    # ---------- DRAW ----------
    if best_bbox is not None:
        color = (0, 0, 255) if check_violation(best_bbox) else (0, 255, 0)
        cv2.rectangle(frame,
                      (best_bbox[0], best_bbox[1]),
                      (best_bbox[2], best_bbox[3]),
                      color, 2)

    # ---------- UI ----------
    cv2.putText(frame, f"Score: {score}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    if not test_started:
        cv2.putText(frame, "PRESS 'S' TO START TEST",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

    cv2.imshow("RTO AI Driving Test System", frame)

    key = cv2.waitKey(frame_delay) & 0xFF

    if key == ord('s') and not test_started:
        test_started = True
        inside_lane = False
        last_bbox = None
        violation_frame_count = 0
        print("Test started manually")

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
