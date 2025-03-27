import cv2
import mediapipe as mp
import numpy as np
import math
import os
import pygame
import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

from collections import deque

index_positions = deque(maxlen=10)  # store last few index tip positions
prev_angle = None
angle_accumulated = 0
volume_change_threshold = 30  # degrees

prev_hand_angle = None
hand_angle_accumulated = 0
twist_threshold = 40           # degrees — increase for stability
twist_cooldown = 0.6           # seconds
last_twist_time = 0  # degrees before volume changes

load_dotenv()

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-modify-playback-state user-read-playback-state"
))

print("Starting in 1 second... Get your hand ready.")
time.sleep(1)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

smoothed_volume = 50
alpha = 0.2
last_toggle_time = 0
cooldown_seconds = 1.5
is_paused = False

last_hand_x = None
swipe_start_time = 0
swipe_cooldown = 2.0
last_swipe_time = 0
toggle_cooldown = 4.0
last_shuffle_toggle = 0

# New:
dx_accumulated = 0
min_swipe_distance = 60  # Increase if needed (50–80 is typical)
min_swipe_time = 0.25

# SWIPE CHECKs

swipe_locked = False
swipe_lock_time = 0

prev_distance = 0
dead_zone_threshold = 5
song_title = ""


def set_volume_mac(vol_percent):
    vol_percent = max(0, min(100, int(vol_percent)))
    os.system(f"osascript -e 'set volume output volume {vol_percent}'")


def get_active_device_id():
    devices = sp.devices()
    for d in devices['devices']:
        if d['is_active']:
            return d['id']
    return None


def fingers_up(lm_list):
    fingers = []
    fingers.append(lm_list[4][0] > lm_list[3][0])
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(lm_list[tip][1] < lm_list[tip - 2][1])
    return fingers


def get_hand_twist_angle(lm_list):
    x1, y1 = lm_list[5]    # base of index finger
    x2, y2 = lm_list[17]   # base of pinky finger

    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle


'''
def is_hand_horizontal(lm_list, angle_threshold=30):
    x1, y1 = lm_list[5]   # index finger base
    x2, y2 = lm_list[17]  # pinky base

    dx = x2 - x1
    dy = y2 - y1

    angle = abs(math.degrees(math.atan2(dy, dx)))
    return angle < angle_threshold or angle > (180 - angle_threshold)
'''


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


while True:
    current_time = time.time()
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        # 'Left' or 'Right'
        hand_label = results.multi_handedness[0].classification[0].label
        landmarks = results.multi_hand_landmarks[0]
        lm_list = []
        for id, lm in enumerate(landmarks.landmark):
            h, w, _ = frame.shape
            lm_list.append((int(lm.x * w), int(lm.y * h)))

        # GESTURE VECTORS
        fingers = fingers_up(lm_list)
        open_hand = fingers[1:] == [True, True, True, True]
        closed_hand = fingers[1:] == [False, False, False, False]
        two_fingers_extended = fingers == [False, True, True, False, False]
        three_fingers_extended = fingers == [False, True, True, True, False]
        shuffle_gesture = fingers == [True, False, False, False, True]

        '''
        # === Doorknob twist volume control with cooldown ===
        hand_angle = get_hand_twist_angle(lm_list)
        current_time = time.time()

        if prev_hand_angle is not None:
            angle_diff = hand_angle - prev_hand_angle

            # Normalize angle_diff to [-180, 180]
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            hand_angle_accumulated += angle_diff

            # Trigger only if enough twist and cooldown has passed
            if hand_angle_accumulated > twist_threshold and current_time - last_twist_time > twist_cooldown:
                smoothed_volume = min(100, smoothed_volume + 5)
                set_volume_mac(smoothed_volume)
                print("↻ Twist right → Volume UP")
                hand_angle_accumulated = 0
                last_twist_time = current_time

            elif hand_angle_accumulated < -twist_threshold and current_time - last_twist_time > twist_cooldown:
                smoothed_volume = max(0, smoothed_volume - 5)
                set_volume_mac(smoothed_volume)
                print("↺ Twist left → Volume DOWN")
                hand_angle_accumulated = 0
                last_twist_time = current_time

        prev_hand_angle = hand_angle
        '''

        if shuffle_gesture and current_time - last_shuffle_toggle > toggle_cooldown:
            sp.shuffle(True)
            print("Shuffle Mode: ON")
            last_shuffle_toggle = current_time

        if two_fingers_extended and current_time - last_swipe_time > swipe_cooldown:
            sp.next_track()
            print("Peace → NEXT track")
            last_swipe_time = current_time

        if three_fingers_extended and current_time - last_swipe_time > swipe_cooldown:
            sp.previous_track()
            print("Three → PREVIOUS track")
            last_swipe_time = current_time

        # Swipe gesture logic
        # SWIPE GESTURE (right, next only)

        '''
        if three_fingers_extended:
            current_x = lm_list[9][0]  # center of palm

            if last_hand_x is None:
                last_hand_x = current_x
                swipe_start_time = current_time
                dx_accumulated = 0
            else:
                dx = current_x - last_hand_x
                dx_accumulated += dx
                last_hand_x = current_x

                swipe_duration = current_time - swipe_start_time

                if swipe_duration > min_swipe_time and current_time - last_swipe_time > swipe_cooldown:
                    if dx_accumulated > min_swipe_distance:
                        sp.next_track()
                        print("🎵 Swiped right with 3 fingers → NEXT track")
                        last_swipe_time = current_time
                        last_hand_x = None
                        dx_accumulated = 0
        else:
            last_hand_x = None
            dx_accumulated = 0
        '''
        # Play/Pause gestures
        if open_hand and is_paused and current_time - last_toggle_time > cooldown_seconds:
            device_id = get_active_device_id()
            if device_id:
                try:
                    sp.start_playback(device_id=device_id)
                except Exception as e:
                    print("Error resuming playback:", e)
            else:
                print("No active Spotify device found.")
            is_paused = False
            last_toggle_time = current_time

        elif closed_hand and not is_paused and current_time - last_toggle_time > cooldown_seconds:
            sp.pause_playback()
            is_paused = True
            last_toggle_time = current_time

        # Get current song info
        current = sp.current_playback()
        if current and current["is_playing"] and current["item"]:
            track = current["item"]["name"]
            artist = current["item"]["artists"][0]["name"]
            song_title = f"{track} - {artist}"

        # Volume control
        if not is_paused:
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            cv2.circle(frame, thumb_tip, 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, index_tip, 8, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, thumb_tip, index_tip, (255, 0, 255), 2)

            distance = get_distance(thumb_tip, index_tip)
            if abs(distance - prev_distance) < dead_zone_threshold:
                distance = prev_distance
            prev_distance = distance

            if distance < 30:
                smoothed_volume = 0
            else:
                raw_volume = np.interp(distance, [30, 120], [0, 100])
                smoothed_volume = alpha * raw_volume + \
                    (1 - alpha) * smoothed_volume

            set_volume_mac(smoothed_volume)

        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        last_hand_x = None  # Reset swipe if no hand detected

    # === UI Rendering ===
    vol_bar_y = np.interp(smoothed_volume, [0, 100], [400, 150])
    cv2.rectangle(frame, (30, 150), (70, 400), (220, 220, 220), 2)
    cv2.rectangle(frame, (30, int(vol_bar_y)),
                  (70, 400), (0, 220, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(smoothed_volume)}%', (25, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    status_text = "PAUSED" if is_paused else (
        "MUTED" if smoothed_volume == 0 else "PLAYING")
    text_size = cv2.getTextSize(
        status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, status_text, (text_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    title_size = cv2.getTextSize(
        song_title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    title_x = (frame.shape[1] - title_size[0]) // 2
    cv2.putText(frame, song_title, (title_x, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Uncomment this to view the camera feed
    # cv2.imshow("Gesture Music Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
