import os
from ultralytics import YOLO
import json
from PIL import Image
import time
from datetime import datetime
from pylibdmtx.pylibdmtx import decode
import pandas as pd
import cv2

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Load models
general_model_path = 'general_model.pt'
general_model = YOLO(general_model_path)

sandwich_model_path = 'sandwich_classifier.pt'
sandwich_model = YOLO(sandwich_model_path)

# Load lookup table
product_list = pd.read_excel('product_list.xlsx')

def process_sandwich(sandwich_image):
    '''
    Runs sandwich classifier on an image
    input: sandwich_image: Image
    output: result: dict
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

def process_data_matrix(data_matrix_image):
    '''
    Runs data matrix decoder on an image
    input: data_matrix_image: Image
    output: result: str
    '''

    # decode the data matrix
    result = decode(data_matrix_image)
    return str(result[0]).split("'")


def validate(item):
    '''
    Validates an item, prompting user to correct if necessary
    input: item: tuple of (item JSON object, Image)
    output: item: tuple of (item JSON object, Image)
    '''

    # Check for if item is "SANDWICH", if so, it is not correctly classified and requires user input
    if item[0].item == "SANDWICH":
        img = item[1]
        img.show()

        print("Could not identify sandwich, please enter the sandwich name:")
        item[0].item = input()
        item[0].user_input = True

        row = product_list[product_list['ProductDescription'] == item[0].item]
        if not row.empty:
            item[0].SKU = row['SKU'].values[0]
            item[0].barcode = row['Barcode'].values[0]
            item[0].lookupcode = row['LookupCode'].values[0]
    
    return item
    
def process_image(image):
    '''
    Processes an image, running it through the general model, sandwich classifier, and data matrix decoder
    input: image: Image
    output: result: dict
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
        if item_name.lower() == "sandwich":
            sandwich_result = process_sandwich(item_image)
            if sandwich_result is not None:
                item[0].item = sandwich_result["item"]
                item[0].sandwich_confidence = sandwich_result["confidence"]
        
        # Run the data matrix detector
        data_matrix_result = process_data_matrix(item_image)
        item[0].data_matrix = data_matrix_result

        # Look up the item in the product list
        row = product_list[product_list['ProductDescription'] == item[0].item]
        if not row.empty:
            item[0].SKU = row['SKU'].values[0]
            item[0].barcode = row['Barcode'].values[0]
            item[0].lookupcode = row['LookupCode'].values[0]

        items.append(item)

    # Validate the items
    for item in items:
        item[0] = validate(item)
    
    # Create the final JSON object
    result = {
        "card_id": "1",
        "date_created": datetime.now().isoformat(),
        "items": [item[0] for item in items]
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
        cv2.destroyAllWindows()

        if image is not None:
            result = process_image(image)

            print(json.dumps(result, indent=4))

            with open('result.json', 'w') as f:
                json.dump(result, f, indent=4)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"Time taken: {time.time() - start_time:.2f}s")