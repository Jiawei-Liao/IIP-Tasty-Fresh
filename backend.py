import os
from ultralytics import YOLO
import json
from PIL import Image
import time
from datetime import datetime
from pylibdmtx.pylibdmtx import decode
import pandas as pd
import cv2
import torch
from typing import Tuple, Dict, List, Optional

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dummy_image = torch.zeros((1, 3, 640, 640)).to(device)

# Load models with error handling
try:
    general_model = YOLO('general_model.pt').to(device)
    sandwich_model = YOLO('sandwich_classifier.pt').to(device)
except Exception as e:
    print(f"Error loading models: {e}")

# Load lookup table with error handling
try:
    product_list = pd.read_excel('product_list.xlsx')
except Exception as e:
    print(f"Error loading product list: {e}")

def process_sandwich(sandwich_image: Image.Image) -> Optional[Tuple[str, float]]:
    """Runs sandwich classifier on an image"""
    try:
        sandwich_results = sandwich_model(sandwich_image)
        if sandwich_results[0].probs is not None:
            top_class_index = sandwich_results[0].probs.top1
            top_class_confidence = sandwich_results[0].probs.top1conf.item()
            sandwich_name = sandwich_results[0].names[top_class_index]
            return (sandwich_name, top_class_confidence)
    except Exception as e:
        print(f"Error processing sandwich: {e}")
    return None

def process_data_matrix(data_matrix_image: Image.Image) -> Optional[Tuple[str, str]]:
    """Runs data matrix decoder on an image"""
    try:
        result = decode(data_matrix_image, max_count=1, threshold=50, min_edge=20, max_edge=60)
        if result:
            barcode = str(result[0]).split('\'')
            item = product_list.loc[product_list['barcode'] == barcode, 'item'].values
            return (item[0] if len(item) > 0 else None, barcode)
    except Exception as e:
        print(f"Error decoding data matrix: {e}")
    return None

def detect(image: Image.Image) -> List[Dict]:
    """Detects items in an image"""
    general_results = general_model(image)
    items = []

    for detection in general_results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        item_name = general_results[0].names[int(class_id)]
        item_image = image.crop([x1, y1, x2, y2])

        # Creating an item dictionary instead of tuple
        item_info = {
            'item': item_name,
            'confidence': confidence,
            'sandwich_item': None,
            'sandwich_confidence': None,
            'data_matrix_item': None,
            'data_matrix': None,
            'SKU': None,
            'barcode': None,
            'lookupcode': None,
            'user_input': False,
            'bbox': [x1, y1, x2, y2],
            'price': '1'
        }

        # Run detection submodules
        if item_name == 'SANDWICH':
            result = process_sandwich(item_image)
            if result:
                item_info['sandwich_item'], item_info['sandwich_confidence'] = result
                # Overwrite the 'item' with the detected sandwich item
                item_info['item'] = result[0]  # sandwich name

        # Check for data matrix in any item, including sandwiches
        ''' Removing this temporaryly as data matrix reader is too slow for now '''
        # TODO: add data matrix object detection + threshold to just black and white
        data_matrix_result = process_data_matrix(item_image)
        #data_matrix_result = None
        if data_matrix_result:
            item_info['data_matrix_item'], item_info['barcode'] = data_matrix_result
            # Overwrite the 'item' with the data matrix item
            item_info['item'] = data_matrix_result[0]  # item name from data matrix

        items.append(item_info)

    return items
