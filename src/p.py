import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import re
import threading
import queue
import io

# New Audio Dependencies
from gtts import gTTS
import pygame

# Block TensorFlow GPU noise for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from textblob import TextBlob
from PIL import ImageFont, ImageDraw, Image

# TensorFlow/Keras elements
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Audio Engine Setup (Cloud gTTS + In-Memory Playback) ---
audio_queue = queue.Queue()
pygame.mixer.init()

def tts_worker():
    """Background queue worker to handle cloud TTS without halting the camera."""
    while True:
        text = audio_queue.get()
        if text is None: break  # Failsafe shutdown
        
        try:
            # 1. Request audio generation from Google
            tts = gTTS(text=text, lang='en', slow=False)
            
            # 2. Save directly to RAM to prevent Windows file-lock crashes
            audio_data = io.BytesIO()
            tts.write_to_fp(audio_data)
            audio_data.seek(0)
            
            # 3. Play the audio buffer
            pygame.mixer.music.load(audio_data)
            pygame.mixer.music.play()
            
            # 4. Wait for the audio stream to finish before processing the next request
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"[Audio Subsystem Error]: Internet connection required or API failure. Details: {e}")
            
        audio_queue.task_done()

# Initialize the background audio processor
threading.Thread(target=tts_worker, daemon=True).start()

# --- Configuration & Model Selection ---
print("--- Sign Language Recognition System ---")

# Dynamically resolve the project root (one directory up from 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

options = {
    "1": os.path.join(BASE_DIR, "model", "indian_sign_language"),
    "2": os.path.join(BASE_DIR, "model", "american_sign_language"),
    "3": os.path.join(BASE_DIR, "model", "hand gestures"),
    "4": os.path.join(BASE_DIR, "model", "bengali_sign_language")
}

choice = input("Enter choice (1: ISL, 2: ASL, 3: Gestures, 4: BSL): ").strip()
model_dir = options.get(choice)

if not model_dir:
    raise ValueError("❌ Invalid choice!")

# Load model and label encoder
model_path = os.path.join(model_dir, "gesture_lstm_model.h5")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError(f"❌ Assets not found in verified path: {model_dir}")

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
FONT_PATH = os.path.join(BASE_DIR, "font", "bengali.ttf")
font = ImageFont.truetype(FONT_PATH, 32) if os.path.exists(FONT_PATH) else ImageFont.load_default()

# --- Helper Functions ---
def get_combined_landmarks(results):
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_info.classification[0].label
            landmarks = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
            if label == 'Left': left_hand = landmarks
            elif label == 'Right': right_hand = landmarks
    return left_hand + right_hand

def process_final_sentence(text):
    """Refined post-processing using Regex and TextBlob."""
    clean_text = re.sub(r'(.)\1{2,}', r'\1', text)
    corrected = TextBlob(clean_text).correct()
    return str(corrected)

def is_hand_static(current, previous, threshold=0.02):
    if previous is None: return False
    return np.linalg.norm(np.array(current) - np.array(previous)) < threshold

# --- Main Loop ---
print("--- Initializing Webcam ---")
cap = cv2.VideoCapture(0)
sentence = ""
last_prediction = ""
last_time = time.time()
prediction_delay = 1.5 
prev_landmarks = None

print("✅ System Ready.")
print("Controls: 'q' to quit | 'c' to process & speak | 'r' to clear text")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        input_row = get_combined_landmarks(results)
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Prediction Logic
        if is_hand_static(input_row, prev_landmarks):
            if (time.time() - last_time) > prediction_delay:
                X_input = np.expand_dims(np.array(input_row), axis=(0, 1))
                preds = model.predict(X_input, verbose=0)
                pred_label = label_encoder.inverse_transform([np.argmax(preds)])[0]

                if pred_label != last_prediction:
                    if pred_label == "SPACE": sentence += " "
                    elif pred_label == "BACKSPACE": sentence = sentence[:-1]
                    else: sentence += pred_label
                    
                    last_prediction = pred_label
                    last_time = time.time()
        
        prev_landmarks = input_row.copy()

    # --- UI Rendering ---
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    draw.text((10, 20), f"Detected: {last_prediction}", font=font, fill=(0, 255, 0))
    draw.rectangle([(0, 420), (640, 480)], fill=(0, 0, 0))
    draw.text((10, 430), f"Sentence: {sentence}", font=font, fill=(255, 255, 255))

    final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Sign Language Translator", final_frame)

    # --- Key Operations Mapping ---
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'): 
        sentence = process_final_sentence(sentence)
        if sentence.strip():  
            # Dispatch string to the Google Cloud Audio Queue
            audio_queue.put(sentence)
    elif key == ord('r'):
        sentence = ""
        last_prediction = ""

# Cleanup
audio_queue.put(None)
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()