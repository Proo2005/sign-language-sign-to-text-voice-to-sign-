import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import warnings

# Suppress FP16 warnings for CPU execution
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

from scipy.signal import resample

class VoiceCaptureEngine:
    def __init__(self):
        # We keep this at 16000 because Whisper REQUIRES 16kHz
        self.target_rate = 16000 
        print("Initializing ASR Model...")
        import whisper
        self.model = whisper.load_model("base")

    def record_audio(self, duration=4, filename="temp_input.wav"):
        device_id = 1 
        device_info = sd.query_devices(device_id, 'input')
        native_rate = int(device_info['default_samplerate'])
        
        print(f"\n[MIC ACTIVE] Recording... Speak clearly.")
        
        audio_data = sd.rec(int(duration * native_rate), 
                            samplerate=native_rate, 
                            channels=1, 
                            dtype='float32',
                            device=device_id)
        sd.wait() 

        # --- NEW: AUDIO NORMALIZATION ---
        # 1. Remove DC Offset (centers the waveform)
        audio_data = audio_data - np.mean(audio_data)
        
        # 2. Peak Normalization (boosts the loudest part to 1.0)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        # 3. Resample
        from scipy.signal import resample
        num_samples = int(len(audio_data) * self.target_rate / native_rate)
        audio_resampled = resample(audio_data, num_samples)

        # Save as high-quality 16-bit PCM
        audio_int16 = (audio_resampled * 32767).astype(np.int16)
        write(filename, self.target_rate, audio_int16)
        
        return filename
    
    
    def transcribe(self, audio_path):
        """Passes the audio file through the Whisper model."""
        # fp16=False prevents warnings if running on a CPU instead of a dedicated GPU
        result = self.model.transcribe(audio_path, fp16=False)
        text = result["text"].strip()
        
        # Purge the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return text

# --- Execution Test ---
if __name__ == "__main__":
    asr = VoiceCaptureEngine()
    
    # Record and transcribe a 5-second clip
    temp_file = asr.record_audio(duration=5)
    transcribed_text = asr.transcribe(temp_file)
    
    print(f"\n--- ASR Result ---")
    print(f"Captured Speech: '{transcribed_text}'")