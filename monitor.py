import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------- Settings ----------
MODEL_PATH = "face_landmarker.task"
SAVE_DIR = "cheating_captures"
HEAD_TURN_LIMIT = 0.20      # how far the nose can move from center (0 = center)
GAZE_LIMIT = 0.10          # how far the iris can move from eye center
SUSPICIOUS_SECONDS = 1.0    # behavior must last this long before saving
COOLDOWN_SECONDS = 2.0      # wait before saving another image 
GRACE_SECONDS = 0.7         # allow short breaks before resetting the timer

os.makedirs(SAVE_DIR, exist_ok=True)

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=2,
)
landmarker = vision.FaceLandmarker.create_from_options(options)


def analyze_face(lm):
    """Return (head_offset, gaze_offset). Values near 0 mean looking at the screen."""
    # Head: nose tip (1) position between the left (234) and right (454) face edges
    left, right, nose = lm[234].x, lm[454].x, lm[1].x
    face_w = max(right - left, 1e-6)
    head_offset = ((nose - left) / face_w) - 0.5

    # Gaze: iris center vs eye corners (right eye: iris 468, corners 33/133)
    #                                    (left eye:  iris 473, corners 362/263)
    def eye_ratio(iris, c1, c2):
        a, b = lm[c1].x, lm[c2].x
        w = max(abs(b - a), 1e-6)
        return ((lm[iris].x - min(a, b)) / w) - 0.5

    gaze_offset = (eye_ratio(468, 33, 133) + eye_ratio(473, 362, 263)) / 2
    return head_offset, gaze_offset


def save_capture(frame, reason):
    name = time.strftime("%Y%m%d_%H%M%S") + f"_{reason}.jpg"
    path = os.path.join(SAVE_DIR, name)
    cv2.imwrite(path, frame)
    print("Saved:", path)


cap = cv2.VideoCapture(0)
suspicious_since = None
last_saved = 0
last_seen_bad = 0
print("Monitoring started. Press Q in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    reason = None
    faces = result.face_landmarks
    if len(faces) == 0:
        reason = "no_face"
    elif len(faces) > 1:
        reason = "multiple_faces"
    else:
        head, gaze = analyze_face(faces[0])
        cv2.putText(frame, f"head={head:.2f} gaze={gaze:.2f}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if abs(head) > HEAD_TURN_LIMIT:
            reason = "head_turned"
        elif abs(gaze) > GAZE_LIMIT:
            reason = "looking_away"

    now = time.time()
    if reason:
        last_seen_bad = now
        if suspicious_since is None:
            suspicious_since = now
        elapsed = now - suspicious_since
        cv2.putText(frame, f"WARNING: {reason}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if elapsed >= SUSPICIOUS_SECONDS and now - last_saved >= COOLDOWN_SECONDS:
            save_capture(frame, reason)
            last_saved = now
    else:
        if suspicious_since is not None and now - last_seen_bad > GRACE_SECONDS:
            suspicious_since = None
        
        cv2.putText(frame, "OK", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    cv2.imshow("Interview Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()