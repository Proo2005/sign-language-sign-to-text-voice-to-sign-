import asyncio
import websockets
import json
import os
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
            "INTENT": "anim_idle.png",
            "NAME": "anim_name.png",
            "PLEASE": "anim_please.png",
            "THANK": "anim_thank.png"
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
        print("\n[CLIENT CONNECTED] Ready to receive browser audio.")
        try:
            # This loop keeps the connection alive and waits for the browser
            async for message in websocket:
                if isinstance(message, bytes):
                    print("-> Received audio packet from browser. Processing...")
                    
                    temp_file = "temp_browser_audio.webm"
                    with open(temp_file, "wb") as f:
                        f.write(message)
                    
                    spoken_text = await asyncio.to_thread(self.asr.transcribe_file, temp_file)
                    
                    if not spoken_text:
                        print("-> No speech detected.")
                        await websocket.send(json.dumps({"status": "no_speech"}))
                        continue
                        
                    print(f"-> Transcribed: '{spoken_text}'")
                    gloss_array, confidence = self.nlp.translate_to_unified_gloss(spoken_text)
                    print(f"-> Semantic Match: {gloss_array} (Confidence: {confidence:.2f})")
                    
                    json_payload = self.generate_engine_payload(gloss_array)
                    await websocket.send(json_payload)
                    print("-> Payload transmitted to frontend.")
                    
        except websockets.exceptions.ConnectionClosed:
            print("[CLIENT DISCONNECTED]")

async def main():
    server = AvatarWebSocketServer()
    async with websockets.serve(server.run_capture_cycle, "localhost", 8765, ping_interval=None):
        print("\nWebSocket Server running on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())