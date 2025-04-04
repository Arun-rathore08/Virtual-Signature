import cv2 as cv
import numpy as np
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape

# Black canvas
signature_canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
drawing_enabled = True
last_save_time = 0
last_clear_time = 0

# Wave detection variables
wave_buffer = []
wave_buffer_size = 10  
wave_threshold = 100  
wave_count = 0  
wave_required = 2
wave_reset_time = 1 

def is_canvas_empty(canvas):
    return np.all(canvas == 0)  # Returns True if the canvas is fully black

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            pinky_x, pinky_y = int(landmarks[20].x * w), int(landmarks[20].y * h)
            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)

            # Detect Open Hand (Wave)
            fingers_extended = sum(landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20]) >= 3

            # Detect Fist (No Clearing)
            fist_detected = sum(landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20]) == 0

            if fist_detected:
                drawing_enabled = False  
            else:
                drawing_enabled = True

            # Victory Sign -> Save Signature
            victory_sign = (
                landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y
            )
            if victory_sign and time.time() - last_save_time > 2:
                cv.imwrite("signature.png", signature_canvas)
                print("✅ Signature saved as 'signature.png'")
                last_save_time = time.time()

            # Wave Gesture Detection (Clear After 3 Waves)
            if fingers_extended and not fist_detected:
                wave_buffer.append(index_x)  
                if len(wave_buffer) > wave_buffer_size:
                    wave_buffer.pop(0)

                if len(wave_buffer) == wave_buffer_size:
                    motion_range = max(wave_buffer) - min(wave_buffer)

                    if motion_range > wave_threshold and time.time() - last_clear_time > wave_reset_time:
                        wave_count += 1
                        print(f"🔄 Wave detected! Count: {wave_count}/{wave_required}")
                        wave_buffer.clear()
                        last_clear_time = time.time()  

                if wave_count >= wave_required and not is_canvas_empty(signature_canvas):
                    print("🧼 3 Waves Detected - Clearing Screen!")
                    signature_canvas.fill(0)
                    wave_count = 0  

            # Erase with thumb & index touch
            if np.hypot(index_x - thumb_x, index_y - thumb_y) < 30:
                cv.circle(signature_canvas, (index_x, index_y), 20, (0, 0, 0), -1)

            # Draw with index up & middle down
            if drawing_enabled and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y:
                if prev_x is not None and prev_y is not None:
                    cv.line(signature_canvas, (prev_x, prev_y), (index_x, index_y), (255, 255, 255), 3)  # White drawing
                prev_x, prev_y = index_x, index_y
            else:
                prev_x, prev_y = None, None

    cv.imshow("Signature Output", signature_canvas)
    frame = cv.addWeighted(frame, 1, signature_canvas, 0.3, 0)  
    cv.imshow("Hand Gesture Control", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
