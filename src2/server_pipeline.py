import asyncio
import websockets
import json
from voice_engine import VoiceCaptureEngine
from semantic_engine import SemanticGlossMapper

class AvatarWebSocketServer:
    def __init__(self):
        print("Initializing Backend Services...")
        self.asr = VoiceCaptureEngine()
        self.nlp = SemanticGlossMapper()
        self.animation_map = {
            "HOW": "anim_how.png",
            "YOU": "anim_you.png",
            "BOOK": "anim_book.png",
            "MY": "anim_possessive_chest.png",
            "UNKNOWN": "anim_shrug.png",
            "INTENT": "anim_idle.png"
        }

    def generate_engine_payload(self, gloss_sequence):
        payload = []
        for i, gloss in enumerate(gloss_sequence):
            anim_file = self.animation_map.get(gloss, "anim_idle.png")
            payload.append({
                "sequence_index": i,
                "gloss_id": gloss,
                "animation_asset": anim_file,
                "blend_in_ms": 150,
                "playback_speed": 1.0
            })
        return json.dumps({"status": "success", "animations": payload})

    async def run_capture_cycle(self, websocket):
        """Continuously listens and transmits data to the connected client."""
        print("\n[CLIENT CONNECTED] Ready to receive audio triggers.")
        try:
            while True:
                # In a production app, the browser would send the audio buffer.
                # For this prototype, we trigger the local microphone loop.
                print("\nPress ENTER in the terminal to speak (or Ctrl+C to quit)...")
                await asyncio.to_thread(input) 
                
                temp_audio = await asyncio.to_thread(self.asr.record_audio, 4)
                spoken_text = await asyncio.to_thread(self.asr.transcribe, temp_audio)
                
                if not spoken_text:
                    print("-> [DIAGNOSTIC] No speech detected by Whisper. Verify microphone gain.")
                    continue
                    
                print(f"-> Transcribed: '{spoken_text}'")
                gloss_array, _ = self.nlp.translate_to_unified_gloss(spoken_text)
                
                json_payload = self.generate_engine_payload(gloss_array)
                
                # Transmit the payload to the web browser
                await websocket.send(json_payload)
                print("-> Payload transmitted to frontend.")
                
        except websockets.exceptions.ConnectionClosed:
            print("[CLIENT DISCONNECTED]")

async def main():
    server = AvatarWebSocketServer()
    # Start the WebSocket server on port 8765
    async with websockets.serve(server.run_capture_cycle, "localhost", 8765):
        print("\nWebSocket Server running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())