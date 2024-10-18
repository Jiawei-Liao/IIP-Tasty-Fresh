import os
from ultralytics import YOLO
import json
from PIL import Image
import time
from datetime import datetime
from pylibdmtx.pylibdmtx import decode
import pandas as pd
import cv2
from typing import List, Tuple, Dict

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Load models
general_model_path = 'general_model.pt'
general_model = YOLO(general_model_path)

sandwich_model_path = 'sandwich_classifier.pt'
sandwich_model = YOLO(sandwich_model_path)

# Load lookup table
product_list = pd.read_excel('product_list.xlsx')

def process_sandwich(sandwich_image: Image) -> Dict[str, float]:
    '''
    Runs sandwich classifier on an image
    input: 
        sandwich_image: image of a sandwich
    output: 
        JSON of the sandwich name and confidence
    '''

    # Run image through sandwich classifier
    sandwich_results = sandwich_model(sandwich_image)

    # If there is a prediction, return the sandwich name and confidence
    if sandwich_results[0].probs is not None:
        top_class_index = sandwich_results[0].probs.top1
        top_class_confidence = sandwich_results[0].probs.top1conf.item()
        sandwich_name = sandwich_results[0].names[top_class_index]

        return {
            "item": sandwich_name,
            "confidence": top_class_confidence
        }
    else:
        return None

def process_data_matrix(data_matrix_image: Image) -> str:
    '''
    Runs data matrix decoder on an image
    input:
        data_matrix_image: image of a data matrix
    output:
        decoded data matrix
    '''

    # decode the data matrix
    result = decode(data_matrix_image)
    return str(result[0]).split("'") if result else None

def validate(item: Tuple[Dict, Image]) -> Dict:
    '''
    Validates an item, prompting user to correct if necessary
    input: 
        item: Tuple of item JSON object and image
    output:
        Tuple of validated item JSON object and image
    '''
    image = item[1]
    item = item[0]

    # Check for if item is "SANDWICH", if so, it is not correctly classified and requires user input
    if item['item'] == "SANDWICH":
        image.show()

        print("Could not identify sandwich, please enter the sandwich name:")
        item['item'] = input()
        item['user_input'] = True

        row = product_list[product_list['ProductDescription'] == item['item']]
        if not row.empty:
            item['SKU'] = row['SKU'].values[0]
            item['barcode'] = row['barcode'].values[0]
            item['lookupcode'] = row['lookupcode'].values[0]
    # Check if item matches barcode
    elif item['data_matrix'] is not None:
        barcode = item['data_matrix'][1]
        row = product_list[product_list['barcode'] == barcode]
        if not row.empty:
            item['SKU'] = row['SKU'].values[0]
            item['barcode'] = row['barcode'].values[0]
            item['lookupcode'] = row['lookupcode'].values[0]
        else:
            img = item[1]
            img.show()

            print("Barcode does not match any product, please enter the product name:")
            item['item'] = input()
            item['user_input'] = True
    
    return item
    
def process_image(image: Image) -> Dict:
    '''
    Processes an image, running it through the general model, sandwich classifier, and data matrix decoder
    input: 
        image: image to process to identify items
    output:
        JSON object of the items identified in the image
    '''

    # Run general model on image
    general_results = general_model(image)

    items = []
    for detection in general_results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        item_name = general_results[0].names[int(class_id)]
        item_image = image.crop([x1, y1, x2, y2])

        # (Item JSON object, Image)
        item = (
            {
                "item": item_name,
                "confidence": confidence,
                "sandwich_confidence": None,
                "data_matrix": None,
                "SKU": None,
                "barcode": None,
                "lookupcode": None,
                "user_input": False
            },
            item_image
        )

        # If the item is a sandwich, crop the image and run it through the sandwich classifier
        if item_name == "SANDWICH":
            sandwich_result = process_sandwich(item_image)
            if sandwich_result is not None:
                item[0]['item'] = sandwich_result["item"]
                item[0]['sandwich_confidence'] = sandwich_result["confidence"]
        
        # Run the data matrix detector
        data_matrix_result = process_data_matrix(item_image)

        item[0]['data_matrix'] = data_matrix_result

        items.append(item)

    if len(items) == 0:
        return None
    
    # Validate the items
    items = [validate(item) for item in items]
    
    # Create the final JSON object
    result = {
        "card_id": "1",
        "date_created": datetime.now().isoformat(),
        "items": items
    }

    return result

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    image = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            image = frame

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Finding items...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = process_image(Image.fromarray(image))

            if result is None:
                print("No items found.")
                return
            
            print(json.dumps(result, indent=4))

            with open('result.json', 'w') as f:
                json.dump(result, f, indent=4)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time:.2f}s")