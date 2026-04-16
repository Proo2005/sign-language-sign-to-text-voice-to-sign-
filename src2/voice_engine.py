import whisper
import os
import warnings

# Suppress FP16 warnings for CPU execution
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

class VoiceCaptureEngine:
    def __init__(self):
        print("Initializing ASR Model...")
        # Using the small model for better accuracy
        self.model = whisper.load_model("small")

    def transcribe_file(self, audio_path):
        """Passes the web audio file directly through the Whisper model."""
        try:
            # fp16=False prevents warnings if running on a CPU instead of a GPU
            result = self.model.transcribe(audio_path, fp16=False)
            text = result["text"].strip()
            return text
        except Exception as e:
            print(f"Whisper Transcription Error: {e}")
            return ""
        finally:
            # Purge the temporary audio file sent by the browser to save space
            if os.path.exists(audio_path):
                os.remove(audio_path)