from fastapi import FastAPI, WebSocket, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from typing import List
from cart_item import CartItem
from detect import detect
from PIL import Image
import pandas as pd
import json
import datetime
import os
from yolov8_person_detector import yolov8_person_detector
from rembg import remove
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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

"""
Websocket endpoint for real-time detection
"""
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

"""
Get endpoint to fetch product list
"""
@app.get("/products")
async def get_products():
    try:
        # Load product list
        df = pd.read_excel("product_list.xlsx")

        # Fill NaN values with empty string
        df = df.fillna("")
        
        # Convert to dictionary
        products = df.to_dict('records')
        
        # Return as JSON
        return JSONResponse(content=products)
    
    except Exception as e:
        print(f"Error loading product list: {e}")
        return

# Directory to save transations
os.makedirs("transactions", exist_ok=True)

"""
Saves customer image, with just the customer and removes background
"""
def crop_customer_image(customer_image: bytes):
    # Open the image and convert to RGB
    customer_image = Image.open(io.BytesIO(customer_image)).convert("RGB")
    
    # Use the YOLOv8 model to detect the person
    result = yolov8_person_detector.predict(customer_image)
    
    if result:
        x1, y1, x2, y2, _, _ = result[0].boxes.data[0].tolist()
        customer_image = customer_image.crop((x1, y1, x2, y2))
    
    # Regardless of detection, remove the background
    customer_image = remove(customer_image)
    
    return customer_image

"""
Post endpoint to save transaction data and images
"""
@app.post("/save-transaction")
async def save_transaction(
    itemImage: UploadFile = File(...),
    customerImage: UploadFile = File(...),
    transactionData: UploadFile = File(...)
):
    try:
        # Create transaction directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        transaction_dir = f'transactions/transaction_{timestamp}'
        os.makedirs(transaction_dir, exist_ok=True)

        # Save item image
        item_image_path = f'{transaction_dir}/item.png'
        item_image_content = await itemImage.read()
        with open(item_image_path, 'wb') as f:
            f.write(item_image_content)

        # Save customer image
        customer_image_path = f'{transaction_dir}/customer.png'
        customer_image = await customerImage.read()
        customer_image = crop_customer_image(customer_image)
        customer_image.save(customer_image_path)

        # Save transaction data
        transaction_json = await transactionData.read()
        transaction_data = json.loads(transaction_json)
        
        # Add timestamp to transaction data
        transaction_data['timestamp'] = timestamp
        
        # Save JSON data
        json_path = f'{transaction_dir}/transaction.json'
        with open(json_path, 'w') as f:
            json.dump(transaction_data, f, indent=4)

        return JSONResponse(content={
            "status": "success",
            "message": "Transaction saved successfully",
            "transaction_id": timestamp
        })

    except Exception as e:
        print(f"Error saving transaction: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )