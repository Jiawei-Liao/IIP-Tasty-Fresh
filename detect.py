import os
from ultralytics import YOLO
import json
from PIL import Image
import time
from datetime import datetime
from pylibdmtx.pylibdmtx import decode
import pandas as pd
import cv2
from typing import Tuple, Dict
import argparse
import torch

# os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Load models
dummy_image = torch.zeros((1, 3, 640, 640)).to('cuda')

general_model_path = 'general_model.pt'
general_model = YOLO(general_model_path)
general_model.to('cuda')
general_model(dummy_image)

sandwich_model_path = 'sandwich_classifier.pt'
sandwich_model = YOLO(sandwich_model_path)
sandwich_model.to('cuda')
sandwich_model(dummy_image)

# Load lookup table
product_list = pd.read_excel('product_list.xlsx')

general_items = ['SANDWICH', 'BURRITO']

def process_sandwich(sandwich_image: Image.Image) -> Tuple[str, float]:
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

        return (sandwich_name, top_class_confidence)
    else:
        return None

def process_data_matrix(data_matrix_image: Image.Image) -> str:
    '''
    Runs data matrix decoder on an image
    input:
        data_matrix_image: image of a data matrix
    output:
        decoded data matrix
    '''

    # decode the data matrix
    result = decode(data_matrix_image, max_count=1, threshold=50, min_edge=20, max_edge=60)
    return str(result[0]).split('\'') if result else None

def user_input(image: Image.Image) -> str:
    '''
    Prompts user to input the item name
    input:
        image: image of the item
    output:
        item name
    '''

    image.show()
    print('Could not identify item, please enter the item name:')
    return input()

def validate(item_obj: Tuple[Dict, Image.Image]) -> Dict:
    '''
    Validates an item, prompting user to correct if necessary
    input: 
        item: Tuple of item JSON object and image
    output:
        Tuple of validated item JSON object and image
    '''
    item = item_obj[0]
    image = item_obj[1]

    # Check for if item is 'SANDWICH', if so, it is not correctly classified and requires user input
    if item['item'] in general_items:
        # If both sandwich item and data matrix is found
        if (item['sandwich_item'] and item['data_matrix']):
            barcode = item['data_matrix']
            row = product_list[product_list['barcode'] == barcode]
            if not row.empty:
                data_matrix_item = product_list[item['data_matrix']].values[0]
                # Check if they match
                if data_matrix_item == item['sandwich_item']:
                    item['item'] = item['sandwich_item']
                # Inconsistency found, prompt user to correct
                else:
                    item['item'] = user_input(image)
                    item['user_input'] = True
            else:
                print('${barcode} not found in lookup table')
        # If only sandwich or data matrix is found, use that information
        elif item['sandwich_item']:
            item['item'] = item['sandwich_item']
        elif item['data_matrix']:
            item['item'] = product_list[item['data_matrix']].values[0]
        # If neither is found, prompt user to correct
        else:
            item['item'] = user_input(image)
            item['user_input'] = True
    
    # Check if item matches barcode from data matrix
    elif item['data_matrix'] is not None:
        barcode = item['data_matrix']
        row = product_list[product_list['barcode'] == barcode]
        if not row.empty:
            data_matrix_item = product_list[item['data_matrix']].values[0]
            # Check if item and data matrix item match
            if data_matrix_item == item['item']:
                item['item'] = item['data_matrix']
            # Inconsistency found, prompt user to correct
            else:
                item['item'] = user_input(image)
                item['user_input'] = True
        else:
            print('${barcode} not found in lookup table')

    # Fetch SKU, barcode, and lookup code from lookup table
    row = product_list[product_list['ProductDescription'] == item['item']]
    if not row.empty:
        item['SKU'] = str(row['SKU'].values[0])
        item['barcode'] = str(row['barcode'].values[0])
        item['lookupcode'] = str(row['lookupcode'].values[0])
    else:
        print('Product ${product} not found in lookup table')
    
    return item
    
def process_image(image: Image.Image) -> Dict:
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
                'item': item_name,
                'confidence': confidence,
                'sandwich_item': None,
                'sandwich_confidence': None,
                'data_matrix': None,
                'SKU': None,
                'barcode': None,
                'lookupcode': None,
                'user_input': False
            },
            item_image
        )

        # If the item is a sandwich, crop the image and run it through the sandwich classifier
        if item_name == 'SANDWICH':
            sandwich_item, sandwich_confidence = process_sandwich(item_image)
            if sandwich_item is not None:
                item[0]['sandwich_item'] = sandwich_item
                item[0]['sandwich_confidence'] = sandwich_confidence
        
        # Run the data matrix detector
        # data_matrix_result = process_data_matrix(item_image)
        # item[0]['data_matrix'] = data_matrix_result

        items.append(item)

    if len(items) == 0:
        return None
    
    # Validate the items
    items = [validate(item) for item in items]
    
    # Create the final JSON object
    result = {
        'card_id': '1',
        'date_created': datetime.now().isoformat(),
        'items': items
    }

    return result

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='IIP Detect')

    parser.add_argument('--use_cam', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Enable or disable camera usage (True/False). Default is False.')
    parser.add_argument('--img_path', type=str, default=None,
                    help='Path to image to process. Default is None.')
    
    args = parser.parse_args()

    useCamera = args.use_cam
    img_path = args.img_path

    # Live input
    if useCamera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Error: Could not open camera.')
            return
        
        image = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('Error: Could not read frame.')
                    break

                image = frame

                results = general_model(frame)

                for detection in results[0].boxes.data:
                    x1, y1, x2, y2, confidence, class_id = detection.tolist()
                    item_name = results[0].names[int(class_id)]

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{item_name}: {confidence:.2f}', (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Finding items...')
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows

            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = process_image(Image.fromarray(image))

                if result is None:
                    print('No items found.')
                    return
                
                print(json.dumps(result, indent=4))

                with open('result.json', 'w') as f:
                    json.dump(result, f, indent=4)
    # Image input
    else:
        if img_path is None:
            img_path = 'demo/2.png'
        else:
            img_path = os.path.join('demo', img_path)
        
        image = Image.open(img_path)
        result = process_image(image)

        if result is None:
            print('No items found.')
            return
        
        print(json.dumps(result, indent=4))

        with open('result.json', 'w') as f:
            json.dump(result, f, indent=4)
            
if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Time taken: {time.time() - start_time:.2f}s')