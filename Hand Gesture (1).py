import cv2 #for image processing
import numpy as np
import screen_brightness_control as sbc
import pyautogui
import time #for delays
from ultralytics import YOLO
import math

# --- AUDIO LIBRARY SETUP ---
volume = None
minVol = -65.25
maxVol = 0.0
audio_enabled = False

try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    device = AudioUtilities.GetSpeakers()
    if hasattr(device, 'EndpointVolume'):
        volume = device.EndpointVolume
    else:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    if volume:
        volRange = volume.GetVolumeRange()
        minVol = volRange[0]
        maxVol = volRange[1]
        audio_enabled = True
        print("Audio System Initialized.")
except Exception as e:
    print(f"Audio Warning: {e}")

# --- CONFIGURATION ---
MIN_Y_THRESHOLD = 0.2  # Hand high (20%) = 100% Value
MAX_Y_THRESHOLD = 0.8  # Hand low (80%) = 0% Value

mute_cooldown = 0
screenshot_cooldown = 0
is_muted = False

# --- LOAD MODEL ---
print("Loading YOLOv8 Model...")
model = YOLO('yolov8n-pose.pt')

# --- HELPER FUNCTIONS ---
def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def draw_bar(img, x, y, value, color, label):
    bar_height = 200
    bar_width = 30
    draw_val = max(0, min(100, value))
    filled_height = int(map_value(draw_val, 0, 100, 0, bar_height))
    cv2.rectangle(img, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), 3)
    cv2.rectangle(img, (x, y + bar_height - filled_height), (x + bar_width, y + bar_height), color, cv2.FILLED)
    cv2.putText(img, f'{int(draw_val)}%', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    cv2.putText(img, label, (x, y + bar_height + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
print("System Ready.")
print("Strict Mode Active:")
print(" - 1 Hand  = Adjust Vol/Bright")
print(" - 2 Hands = Check Mute/Screenshot (Vol/Bright Locked)")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Decrease cooldowns
    if mute_cooldown > 0: mute_cooldown -= 1
    if screenshot_cooldown > 0: screenshot_cooldown -= 1

    # Run Inference
    results = model(frame, verbose=False, stream=True, imgsz=320)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0].cpu().numpy()
            
            # Keypoints: 0=Nose, 9=Left Wrist, 10=Right Wrist
            nose_x, nose_y, nose_conf = kpts[0]
            lw_x, lw_y, lw_conf = kpts[9]
            rw_x, rw_y, rw_conf = kpts[10]
            
            # Check presence of hands
            left_present = lw_conf > 0.5
            right_present = rw_conf > 0.5
            
            # --- SCENARIO 1: BOTH HANDS (Special Gestures ONLY) ---
            if left_present and right_present:
                # Provide visual feedback that we are in "Command Mode"
                cv2.putText(frame, "COMMAND MODE", (w//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Check Distance for MUTE
                wrist_distance = math.hypot(rw_x - lw_x, rw_y - lw_y)
                
                if wrist_distance < 60:
                    if mute_cooldown == 0:
                        is_muted = not is_muted
                        mute_cooldown = 30
                        if audio_enabled:
                            try: volume.SetMute(1 if is_muted else 0, None)
                            except: pass
                    cv2.putText(frame, "MUTE TOGGLED", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Check Height for SCREENSHOT
                elif lw_y < nose_y and rw_y < nose_y:
                    if screenshot_cooldown == 0:
                        screenshot_cooldown = 60
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        pyautogui.screenshot(f"screenshot_{ts}.png")
                    cv2.rectangle(frame, (0, 0), (w, h), (255, 255, 255), 10)
                    cv2.putText(frame, "SCREENSHOT", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # --- SCENARIO 2: SINGLE HAND (Individual Control) ---
            else:
                # LEFT HAND ONLY -> BRIGHTNESS
                if left_present:
                    cv2.circle(frame, (int(lw_x), int(lw_y)), 10, (0, 255, 255), cv2.FILLED)
                    norm_y = np.clip(lw_y / h, MIN_Y_THRESHOLD, MAX_Y_THRESHOLD)
                    bright_val = map_value(norm_y, MAX_Y_THRESHOLD, MIN_Y_THRESHOLD, 0, 100)
                    try:
                        sbc.set_brightness(int(bright_val))
                        draw_bar(frame, 50, 150, bright_val, (0, 255, 255), "Bright")
                    except: pass

                # RIGHT HAND ONLY -> VOLUME
                if right_present:
                    cv2.circle(frame, (int(rw_x), int(rw_y)), 10, (255, 0, 0), cv2.FILLED)
                    norm_y = np.clip(rw_y / h, MIN_Y_THRESHOLD, MAX_Y_THRESHOLD)
                    vol_percent = map_value(norm_y, MAX_Y_THRESHOLD, MIN_Y_THRESHOLD, 0, 100)
                    vol_db = map_value(norm_y, MAX_Y_THRESHOLD, MIN_Y_THRESHOLD, minVol, maxVol)
                    
                    if audio_enabled:
                        try:
                            volume.SetMasterVolumeLevel(vol_db, None)
                            draw_bar(frame, w - 80, 150, vol_percent, (255, 0, 0), "Vol")
                        except: pass

    if is_muted:
        cv2.putText(frame, "MUTED", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
