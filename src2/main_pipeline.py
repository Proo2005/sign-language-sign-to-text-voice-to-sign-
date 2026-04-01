import json
from voice_engine import VoiceCaptureEngine
from semantic_engine import SemanticGlossMapper

class AvatarBackendPipeline:
    def __init__(self):
        print("Initializing Backend Services...")
        # 1. Boot up the Microphone & Whisper Engine
        self.asr = VoiceCaptureEngine()
        
        # 2. Boot up the Transformer Translation Engine
        self.nlp = SemanticGlossMapper()
        
        # 3. Define the Engine Animation Map
        # This links the translated glosses to the actual 3D animation files
        self.animation_map = {
            "HOW": "anim_how.fbx",
            "YOU": "anim_you.fbx",
            "BOOK": "anim_book.fbx",
            "MY": "anim_possessive_chest.fbx",
            "NAME": "anim_name.fbx",
            "WHAT": "anim_what.fbx",
            "UNKNOWN": "anim_shrug.fbx",
            "INTENT": "anim_idle.fbx"
        }

    def generate_engine_payload(self, gloss_sequence):
        """Constructs the instruction set for the Unity/Unreal frontend."""
        payload = []
        for i, gloss in enumerate(gloss_sequence):
            anim_file = self.animation_map.get(gloss, "anim_idle.fbx")
            
            # Construct a frame-by-frame blending instruction
            payload.append({
                "sequence_index": i,
                "gloss_id": gloss,
                "animation_asset": anim_file,
                "blend_in_ms": 150,  # Milliseconds to crossfade from previous animation
                "playback_speed": 1.0
            })
            
        return json.dumps({"status": "success", "animations": payload}, indent=2)

    def run_capture_cycle(self):
        """Executes a single listen-translate-export cycle."""
        print("\n=== Starting Capture Cycle ===")
        
        # Step A: Capture & Transcribe
        temp_audio = self.asr.record_audio(duration=4)
        spoken_text = self.asr.transcribe(temp_audio)
        print(f"-> ASR Output: '{spoken_text}'")
        
        if not spoken_text:
            print("-> No speech detected. Aborting cycle.")
            return None

        # Step B: Semantic Translation to Gloss
        gloss_array, confidence = self.nlp.translate_to_unified_gloss(spoken_text)
        print(f"-> NLP Gloss Translation: {gloss_array} (Confidence: {confidence:.2f})")
        
        # Step C: Generate 3D Engine Payload
        json_payload = self.generate_engine_payload(gloss_array)
        print("\n-> Final Payload for Rendering Engine:")
        print(json_payload)
        
        return json_payload

# --- Execution ---
if __name__ == "__main__":
    system = AvatarBackendPipeline()
    
    # Run a single interactive test cycle
    system.run_capture_cycle()