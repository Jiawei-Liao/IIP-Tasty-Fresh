from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from typing import List
from cart_item import CartItem
from detect import detect
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_frame(frame_data: str) -> List[CartItem]:
    # Decode base64 image
    encoded_data = frame_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Process frame with PIL Image
    detections = detect.detect(image)

    return detections

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive frame from client
            frame_data = await websocket.receive_text()

            # Process frame
            detections = await process_frame(frame_data)

            # Send results back
            await websocket.send_json([d.model_dump() for d in detections])
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        await websocket.send_json([])
            
    finally:
        await websocket.close()
