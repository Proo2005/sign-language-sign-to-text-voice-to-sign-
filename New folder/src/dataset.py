import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# === CONFIG ===
GESTURE_NAME = "O"   # Change this to your gesture label
SAMPLES_TO_COLLECT = 500     # Number of samples per gesture
SAVE_DIR = "dataset/indian_sign_language"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Video Capture ===
cap = cv2.VideoCapture(0)
collected = 0
print(f"Starting to  collect data for '{GESTURE_NAME}'...")

def get_landmarks(results):
    landmarks = {
        'Left': None,
        'Right': None
    }
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            lm = []
            for point in hand_landmarks.landmark:
                lm.extend([point.x, point.y, point.z])
            landmarks[label] = lm
    return landmarks

csv_path = os.path.join(SAVE_DIR, f"{GESTURE_NAME}.csv")
with open(csv_path, mode='a', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)

    while collected < SAMPLES_TO_COLLECT:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            landmark_data = get_landmarks(results)

            left = landmark_data['Left']
            right = landmark_data['Right']

            # Fill missing hand with zeros if only one detected
            if left is None:
                left = [0.0] * 63
            if right is None:
                right = [0.0] * 63

            sample = left + right
            sample.append(GESTURE_NAME)

            writer.writerow(sample)
            collected += 1

            # Draw landmarks
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Show frame
        cv2.putText(frame, f"Collecting {GESTURE_NAME}: {collected}/{SAMPLES_TO_COLLECT}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Two-Hand Gesture Recorder", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break  # Press ESC to exit

cap.release()
cv2.destroyAllWindows()
print(f"✅ Collected {collected} samples for gesture '{GESTURE_NAME}' and saved to {csv_path}")
