import cv2
import numpy as np

# ---------------- LOAD ASSETS ----------------
video_path = "test_video.mp4"
safe_lane_mask = cv2.imread("safe_lane_mask.png", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(video_path)

# ---------------- STATE ----------------
inside_lane = True
score = 100

def check_violation(bbox, safe_lane_mask):
    x1, y1, x2, y2 = bbox
    region = safe_lane_mask[y1:y2, x1:x2]
    return (region == 0).any()

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------
    # THIS bbox will later come from YOLO
    # For now, mock / test bbox
    bbox = (600, 300, 760, 430)
    # -----------------------------------------

    violation_now = check_violation(bbox, safe_lane_mask)

    if inside_lane and violation_now:
        score -= 5
        print("Violation detected! Score deducted.")
        inside_lane = False

    elif not violation_now:
        inside_lane = True

    # -----------------------------------------
    # VISUALIZATION
    color = (0, 0, 255) if violation_now else (0, 255, 0)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Violation Check", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
