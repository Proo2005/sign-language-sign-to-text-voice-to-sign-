import speech_recognition as sr

def start_voice_detection():
    r = sr.Recognizer()
    


    mic_index = 20

    try:
        with sr.Microphone(device_index=mic_index) as source:
            print(f"\n[SYSTEM] Trying Microphone Index: {mic_index}")
            
            # Sensitivity adjustment
            r.energy_threshold = 300 
            r.dynamic_energy_threshold = True 
            r.pause_threshold = 0.8  # Kotha bolar majhe pause nileo jate bondho na hoy
            
            print("[READY] Ekhon kotha bolo... (Joore bolo)")
            
            # Listen korar somoy status update
            audio = r.listen(source, timeout=7, phrase_time_limit=12)
            
            print("[INFO] Recognizing...")
            # Google API use kore text convert
            text = r.recognize_google(audio)
            
            print(f"[RESULT] You said: {text}")
            return text

    except sr.WaitTimeoutError:
        print("[ERROR] Timeout: Mic kichui shunte payni. Index change kore dekho.")
    except sr.UnknownValueError:
        print("[ERROR] Google kotha bujhte pareni. Mic-er kachhe eshe bolo.")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")

if __name__ == "__main__":
    result = start_voice_detection()
    
    # Text detected hole glossing-er logic ekhanei add hobe
    if result:
        print(f"\nProceeding with Sign Translation for: {result}")