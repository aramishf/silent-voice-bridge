from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from web.backend.inference import WebASLInference

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SERVE STATIC FILES (Frontend) for Deployment
# Mount 'assets' to serve JS/CSS
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Path to the React "dist" folder (built by Docker)
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# Verify path exists before mounting (avoids errors in local dev without build)
if (FRONTEND_DIST).exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
    
    # Catch-all route to serve React App (index.html)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Allow API routes to pass through (though they are usually defined above)
        if full_path.startswith("data") or full_path.startswith("api"):
            return None # Should handle 404 naturally if API doesn't exist
            
        # Serve index.html for any other route (SPA Handling)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    print(f"WARNING: Frontend dist folder not found at {FRONTEND_DIST}. Running in API-only mode.")

# Initialize Inference Engine
# Paths are relative to where we run the script (likely project root)
MODEL_PATH = "models/checkpoints/best_model.pth"
CONFIG_PATH = "config/config.yaml"
SCALER_PATH = "data/processed/scaler.pkl"

print("Initializing ASL Engine...")
engine = WebASLInference(MODEL_PATH, CONFIG_PATH, SCALER_PATH)
print("ASL Engine Ready!")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive data (can be text or bytes)
            message = await websocket.receive()
            
            # Check if it's text (command) or bytes (video frame)
            if 'text' in message:
                # It's a JSON command
                try:
                    command = json.loads(message['text'])
                    if 'action' in command:
                        print(f"Received command: {command['action']}")
                        engine.handle_command(command['action'])
                        # Return state update
                        await websocket.send_json({
                            "type": "state",
                            "data": {
                                "current_sentence": engine.sentence_builder.get_current_sentence(),
                                "history": engine.sentence_builder.sentence_history
                            }
                        })
                        continue
                except Exception as e:
                    print(f"Error processing command: {e}")
                    continue
            
            elif 'bytes' in message:
                # It's a video frame
                data = message['bytes']
                
                # Process Frame
                result = engine.process_frame(data)
                
                if result:
                    # Send back result
                    await websocket.send_json({
                        "type": "result",
                        "data": result
                    })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
