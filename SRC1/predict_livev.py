import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import re
import warnings

from textblob import TextBlob
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import load_model
from gtts import gTTS
from playsound import playsound

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration & Model Selection ---
print("--- Sign Language Recognition System ---")
options = {
    "1": "model/indian_sign_language",
    "2": "model/american_sign_language",
    "3": "model/hand gestures",
    "4": "model/bengali_sign_language"
}

choice = input("Enter choice (1: ISL, 2: ASL, 3: Gestures, 4: BSL): ").strip()
model_dir = options.get(choice)
if not model_dir:
    raise ValueError("❌ Invalid choice!")

speech_lang = 'bn' if 'bengali' in model_dir.lower() else 'en'
use_gtts = choice == "3"  # Only for hand gestures

# --- Load model and label encoder ---
model_path = os.path.join(model_dir, "gesture_lstm_model.h5")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError(f"❌ Assets not found in {model_dir}")

print("--- Loading AI Models (CPU Mode) ---")
model = load_model(model_path)
label_encoder = joblib.load(encoder_path)
print("✅ Models Loaded Successfully")

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- Bengali Font Setup ---
FONT_PATH = "font/bengali.ttf"
font = ImageFont.truetype(FONT_PATH, 32) if os.path.exists(FONT_PATH) else ImageFont.load_default()

# --- Helper Functions ---
def get_combined_landmarks(results):
    left_hand = [0.0]*63
    right_hand = [0.0]*63
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_info.classification[0].label
            landmarks = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
            if label == 'Left': left_hand = landmarks
            elif label == 'Right': right_hand = landmarks
    return left_hand + right_hand

def process_final_sentence(text):
    clean_text = re.sub(r'(.)\1{2,}', r'\1', text)
    corrected = TextBlob(clean_text).correct()
    return str(corrected)

def speak_gtts(text):
    """gTTS for hand gestures only"""
    text = text.strip()
    if not text:
        return
    try:
        filename = "temp_voice.mp3"
        if os.path.exists(filename):
            os.remove(filename)
        tts = gTTS(text=text, lang='en', slow=False)  # fast speech
        tts.save(filename)
        playsound(filename, block=True)
        os.remove(filename)
    except Exception as e:
        print("gTTS Error:", e)

# --- Main Loop ---
print("--- Initializing Webcam ---")
cap = cv2.VideoCapture(0)

sentence = ""
last_prediction = ""
sequence = []
sequence_length = 30  # Match your LSTM training
confidence_threshold = 0.3

print("✅ System Ready. Press 'c' to speak (gestures only), 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    input_row = None
    if results.multi_hand_landmarks:
        input_row = get_combined_landmarks(results)
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # --- Add to rolling sequence ---
    if input_row is not None:
        sequence.append(input_row)
        if len(sequence) > sequence_length:
            sequence.pop(0)

        if len(sequence) == sequence_length:
            X_input = np.expand_dims(np.array(sequence), axis=0)
            preds = model.predict(X_input, verbose=0)
            pred_index = int(np.argmax(preds))
            confidence = float(np.max(preds))
            pred_label = None
            if confidence > confidence_threshold:
                pred_label = label_encoder.inverse_transform([pred_index])[0]

            if pred_label and pred_label != last_prediction:
                if pred_label == "SPACE":
                    sentence += " "
                elif pred_label == "BACKSPACE":
                    sentence = sentence[:-1]
                else:
                    sentence += pred_label
                last_prediction = pred_label
            sequence = []  # Reset sequence after prediction

    # --- UI Rendering ---
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 20), f"Detected: {last_prediction}", font=font, fill=(0, 255, 0))
    draw.rectangle([(0, 420), (640, 480)], fill=(0, 0, 0))
    draw.text((10, 430), f"Sentence: {sentence}", font=font, fill=(255, 255, 255))
    final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Sign Language Translator", final_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = process_final_sentence(sentence)
        if use_gtts:  # Only speak for gestures
            speak_gtts(sentence)

cap.release()
cv2.destroyAllWindows()