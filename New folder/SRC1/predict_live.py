import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import keyboard
from textblob import TextBlob
import re
import os
from PIL import ImageFont, ImageDraw, Image

# === Choose sign language type ===
print("Choose sign language type:")
print("1. Indian Sign Language")
print("2. American Sign Language")
print("3. Hand Gesture")
print("4. Bengali Sign Language")
choice = input("Enter choice (1/2/3/4): ").strip()

if choice == "1":
    model_dir = "model/indian_sign_language"
elif choice == "2":
    model_dir = "model/american_sign_language"
elif choice == "3":
    model_dir = "model/hand gestures"
elif choice == "4":
    model_dir = "model/bengali_sign_language"
else:
    raise ValueError("❌ Invalid choice! Please select 1, 2, 3, or 4.")

# === Load model and label encoder ===
model_path = os.path.join(model_dir, "gesture_model.pkl")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError(f"❌ Model or encoder not found in '{model_dir}'. Please train first.")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# === Load Bengali font ===
FONT_PATH = "font/bengali.ttf"  # Put your downloaded Bengali font here
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Font not found at {FONT_PATH}. Please download and place it there.")
font = ImageFont.truetype(FONT_PATH, 32)

# === Open webcam ===
cap = cv2.VideoCapture(0)

# Track sentence and timing
sentence = ""
last_prediction = ""
last_time = time.time()
prediction_delay = 1.5  # seconds

# For hand movement detection
prev_landmarks = None
movement_threshold = 0.02  # lower = more strict about static hand


def get_combined_landmarks(results):
    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_info.classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if label == 'Left':
                left_hand = landmarks
            elif label == 'Right':
                right_hand = landmarks

    if left_hand is None:
        left_hand = [0.0] * 63
    if right_hand is None:
        right_hand = [0.0] * 63

    return left_hand + right_hand


def correct_text(sentence):
    # Remove repeated characters 
    sentence = re.sub(r'(.)\1{2,}', r'\1', sentence)

    # Correct grammar/spelling
    blob = TextBlob(sentence)
    corrected = str(blob.correct())

    return corrected


def is_hand_static(current, previous, threshold):
    if previous is None:
        return False
    current = np.array(current)
    previous = np.array(previous)
    diff = np.linalg.norm(current - previous)
    return diff < threshold


while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        input_row = get_combined_landmarks(results)

        if len(input_row) == 126 and is_hand_static(input_row, prev_landmarks, movement_threshold):
            current_time = time.time()

            if (current_time - last_time) > prediction_delay:
                pred = model.predict([input_row])[0]
                pred_label = label_encoder.inverse_transform([pred])[0]

                if pred_label != last_prediction:
                    if pred_label == "SPACE":
                        sentence += "\t"
                    elif pred_label == "R":
                        if len(sentence) > 0:
                            sentence = sentence[:-1]
                    else:
                        sentence += pred_label

                    last_prediction = pred_label
                    last_time = current_time

        prev_landmarks = input_row.copy()

        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Keyboard 'r' key to remove last character manually
    if key == ord('r'):
        if len(sentence) > 0:
            sentence = sentence[:-1]
        last_prediction = ""

    # Convert to PIL for Bengali text rendering
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Draw prediction text
    if last_prediction:
        draw.text((10, 30), f'Prediction: {last_prediction}', font=font, fill=(0, 255, 0))

    # Draw sentence at bottom
    draw.rectangle([(0, 400), (640, 480)], fill=(0, 0, 0))
    draw.text((10, 430), sentence.strip(), font=font, fill=(255, 255, 255))

    # Convert back to OpenCV format
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow("Two-Hand Gesture Recognition", image)

    # Quit key
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()