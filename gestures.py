import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui

# =========================
# Mediapipe setup
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.85,
                       min_tracking_confidence=0.77,
                       model_complexity=1)

cap = cv2.VideoCapture(0)
WIN = "Gesture Media Controller"

# =========================
# Volume setup
# =========================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
vol = cast(interface, POINTER(IAudioEndpointVolume))


def get_vol_scalar():
    return float(vol.GetMasterVolumeLevelScalar())


def set_vol_scalar(s):
    s = max(0.0, min(1.0, float(s)))
    vol.SetMasterVolumeLevelScalar(s, None)


def volume_step(delta=0.05):
    set_vol_scalar(get_vol_scalar() + delta)


# =========================
# Media controls
# =========================
def media_play_pause():
    pyautogui.press("playpause")  # Windows


def media_next():
    pyautogui.press("nexttrack")


def media_prev():
    pyautogui.press("prevtrack")


# =========================
# Helpers
# =========================
def angle_between(p0, p1, p2):
    v1 = (p0[0] - p1[0], p0[1] - p1[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))


# =========================
# Gesture variables
# =========================
ROT_STEP_DEG = 25
VOL_STEP = 0.05
ROT_DEAD_DEG = 5
DEBUG = True
rot_prev_angle = None
rot_accum = 0.0
palm_latched = False
gesture_cooldown = 0
GESTURE_COOLDOWN_FRAMES = 15  # frames cooldown between next/prev triggers

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        pts = [(int(l.x * w), int(l.y * h)) for l in hand.landmark]

        # ----- Finger states -----
        def finger_up(tip, pip):
            return pts[tip][1] < pts[pip][1]

        index_up = finger_up(8, 6)
        middle_up = finger_up(12, 10)
        ring_up = finger_up(16, 14)
        pinky_up = finger_up(20, 18)

        palm_open = index_up and middle_up and ring_up and pinky_up
        palm_closed = not palm_open

        # ----- Play/Pause -----
        if palm_open:
            if not palm_latched:
                media_play_pause()
                palm_latched = True
                if DEBUG:
                    cv2.putText(frame, "Play/Pause", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
        else:
            palm_latched = False

        # ----- Volume -----
        vol_gate = index_up and not middle_up and not ring_up and not pinky_up
        if vol_gate:
            base = pts[5]
            tip = pts[8]
            dx = tip[0] - base[0]
            dy = tip[1] - base[1]
            angle = math.degrees(math.atan2(dy, dx))
            if rot_prev_angle is not None:
                diff = angle - rot_prev_angle
                if diff > 180: diff -= 360
                if diff < -180: diff += 360
                if abs(diff) < ROT_DEAD_DEG: diff = 0.0
                rot_accum += diff
                while rot_accum >= ROT_STEP_DEG:
                    volume_step(+VOL_STEP)
                    rot_accum -= ROT_STEP_DEG
                    if DEBUG:
                        cv2.putText(frame, "Vol +", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                while rot_accum <= -ROT_STEP_DEG:
                    volume_step(-VOL_STEP)
                    rot_accum += ROT_STEP_DEG
                    if DEBUG:
                        cv2.putText(frame, "Vol -", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            rot_prev_angle = angle
            cv2.circle(frame, base, 6, (0, 200, 255), -1)
            cv2.circle(frame, tip, 6, (0, 200, 255), -1)
            cv2.line(frame, base, tip, (0, 200, 255), 2)
        else:
            rot_prev_angle = None
            rot_accum = 0.0

        # ----- Next/Prev Song Gesture -----
        if gesture_cooldown > 0:
            gesture_cooldown -= 1

        # Condition: palm closed, index and pinky up
        if palm_closed and index_up and pinky_up and gesture_cooldown == 0:
            thumb_tip = pts[4]
            wrist = pts[0]
            dx = thumb_tip[0] - wrist[0]

            if dx > 40:  # thumb pointing right
                media_next()
                gesture_cooldown = GESTURE_COOLDOWN_FRAMES
                if DEBUG:
                    cv2.putText(frame, "Next ▶▶", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            elif dx < -40:  # thumb pointing left
                media_prev()
                gesture_cooldown = GESTURE_COOLDOWN_FRAMES
                if DEBUG:
                    cv2.putText(frame, "◀◀ Prev", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        # ----- Debug info -----
        cv2.putText(frame, f"Vol: {int(get_vol_scalar()*100)}%", (w-150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Finger Active: {'Yes' if vol_gate else 'No'}", (w-250, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Window exit
    cv2.imshow(WIN, frame)
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
