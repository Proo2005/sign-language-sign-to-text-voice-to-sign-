import speech_recognition as sr
import cv2 # If using 2D image-based avatar

# 1. Simple Mapping Database
avatar_db = {
    "hello": "wave_gesture.png",
    "fist": "fist_gesture.png",
    "stop": "stop_gesture.png"
}

def run_avatar_pipeline():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)
        
        try:
            # 2. Speech to Text
            text = r.recognize_google(audio).lower()
            print(f"You said: {text}")
            
            # 3. Match and Show Avatar
            if text in avatar_db:
                img = cv2.imread(avatar_db[text])
                cv2.imshow("Avatar", img)
                cv2.waitKey(2000) # Show for 2 seconds
            else:
                print("No avatar gesture found for that word.")
                
        except Exception as e:
            print("Could not understand audio.")

run_avatar_pipeline()