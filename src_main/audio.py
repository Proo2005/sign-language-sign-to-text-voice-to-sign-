import pyttsx3
import threading
import queue # Add this new import

# --- Audio Engine Setup (Worker Thread Method) ---
audio_queue = queue.Queue()

def tts_worker():
    """Dedicated background loop for safe audio execution on Windows."""
    # The engine MUST be initialized inside the thread that uses it
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    while True:
        text = audio_queue.get()
        if text is None: break  # Failsafe to exit thread
        engine.say(text)
        engine.runAndWait()
        audio_queue.task_done()

# Start the audio worker thread immediately in the background
threading.Thread(target=tts_worker, daemon=True).start()