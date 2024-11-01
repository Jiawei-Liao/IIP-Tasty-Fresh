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
from functools import lru_cache
import numpy as np
from threading import Lock

class DetectionBackend:
    def __init__(self):
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize locks
        self.model_lock = Lock()
        
        # Load models
        try:
            print(f"Loading models on {self.device}...")
            self.general_model = YOLO('general_model.pt').to(self.device)
            self.sandwich_model = YOLO('sandwich_classifier.pt').to(self.device)
            
            # Warmup models
            self._warmup_models()
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
            
        # Load product list
        try:
            self.product_list = pd.read_excel('product_list.xlsx')
        except Exception as e:
            print(f"Error loading product list: {e}")
            raise
            
        # Initialize result cache
        self.result_cache = {}
        self.cache_lock = Lock()
        
        print("Backend initialization complete")

    def _warmup_models(self):
        """Warm up models with dummy data to improve initial inference speed"""
        print("Warming up models...")
        dummy_image = torch.zeros((1, 3, 640, 640)).to(self.device)
        with torch.no_grad():
            self.general_model(dummy_image)
            self.sandwich_model(dummy_image)

    @lru_cache(maxsize=100)
    def process_sandwich(self, image_hash: str, image: Image.Image) -> Optional[Tuple[str, float]]:
        """Runs sandwich classifier on an image with caching"""
        try:
            with self.model_lock:
                sandwich_results = self.sandwich_model(image)
                
            if sandwich_results[0].probs is not None:
                top_class_index = sandwich_results[0].probs.top1
                top_class_confidence = sandwich_results[0].probs.top1conf.item()
                sandwich_name = sandwich_results[0].names[top_class_index]
                return (sandwich_name, top_class_confidence)
                
        except Exception as e:
            print(f"Error processing sandwich: {e}")
        return None

    def process_data_matrix(self, image: Image.Image) -> Optional[Tuple[str, str]]:
        """Runs data matrix decoder on an image"""
        try:
            # Convert to grayscale and enhance contrast for better detection
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            enhanced = cv2.equalizeHist(gray)
            
            result = decode(enhanced, max_count=1, threshold=50, min_edge=20, max_edge=60)
            
            if result:
                barcode = str(result[0]).split('\'')
                item = self.product_list.loc[self.product_list['barcode'] == barcode, 'item'].values
                return (item[0] if len(item) > 0 else None, barcode)
                
        except Exception as e:
            print(f"Error decoding data matrix: {e}")
        return None

    def detect(self, image: Image.Image) -> List[Dict]:
        """Detects items in an image with improved performance"""
        try:
            # Convert image to tensor once
            with self.model_lock:
                general_results = self.general_model(image)

            items = []
            for detection in general_results[0].boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection.tolist()
                item_name = general_results[0].names[int(class_id)]
                
                # Crop image for additional processing
                item_image = image.crop([x1, y1, x2, y2])
                
                # Basic item info
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
                
                # Process sandwich if detected
                if item_name == 'SANDWICH':
                    # Create a hash for the cropped image for caching
                    image_hash = hash(item_image.tobytes())
                    result = self.process_sandwich(str(image_hash), item_image)
                    
                    if result:
                        item_info['sandwich_item'], item_info['sandwich_confidence'] = result
                        item_info['item'] = result[0]  # sandwich name

                # Add processed item to list
                items.append(item_info)

            return items
            
        except Exception as e:
            print(f"Error in detect: {e}")
            return []

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear other caches
            self.process_sandwich.cache_clear()
            self.result_cache.clear()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Create singleton instance
detection_backend = DetectionBackend()