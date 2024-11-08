from ultralytics import YOLO
from PIL import Image
#from pylibdmtx.pylibdmtx import decode
import pandas as pd
import cv2
import torch
from typing import Tuple, Dict, List, Optional
import numpy as np
from threading import Lock
from cart_item import CartItem

class Detect:
    def __init__(self):
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize locks
        self.model_lock = Lock()
        
        # Load models
        try:
            print(f"Loading models on {self.device}...")
            self.general_model = YOLO('models/general_model.pt').to(self.device)
            self.sandwich_model = YOLO('models/sandwich_classifier.pt').to(self.device)
            
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
        
        print("Backend initialization complete")

    def _warmup_models(self):
        """Warm up models with dummy data to improve initial inference speed"""
        print("Warming up models...")
        dummy_image = torch.zeros((1, 3, 640, 640)).to(self.device)
        with torch.no_grad():
            self.general_model(dummy_image, verbose=False)
            self.sandwich_model(dummy_image, verbose=False)

    def process_sandwich(self, image: Image.Image) -> Optional[List[str]]:
        """
        Runs sandwich classifier on an image

        args:
            image (Image.Image): Image to process
        returns:
            List[str]: List of top 5 sandwich predictions
        """
        try:
            with self.model_lock:
                sandwich_results = self.sandwich_model(image, verbose=False)

            if sandwich_results[0].probs is not None:
                # Get top 5 predictions by sorting the probabilities in descending order
                top5_indices = sandwich_results[0].probs.top5
                # Extract top 5 names (sorted by confidence)
                top5_predictions = [
                    sandwich_results[0].names[idx] for idx in top5_indices
                ]
                
                return top5_predictions

        except Exception as e:
            print(f"Error processing sandwich: {e}")
        
        return None

    def process_data_matrix(self, image: Image.Image) -> Optional[Tuple[str, str]]:
        """
        Runs data matrix decoder on an image

        args:
            image (Image.Image): Image to process
        returns:
            Tuple[str, str]: Item name and barcode
        """
        try:
            # Convert to grayscale and enhance contrast for better detection
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            enhanced = cv2.equalizeHist(gray)
            
            #result = decode(enhanced, max_count=1, threshold=50, min_edge=20, max_edge=60)
            result = None
            if result:
                barcode = str(result[0]).split('\'')
                item = self.product_list.loc[self.product_list['barcode'] == barcode, 'item'].values
                return (item[0] if len(item) > 0 else None, barcode)
                
        except Exception as e:
            print(f"Error decoding data matrix: {e}")
        return None

    def detect(self, image: Image.Image) -> List[CartItem]:
        """
        Detects items in an image
        
        args:
            image (Image.Image): Image to process
        returns:
            List[CartItem]: List of detected items
        """
        try:
            # Get results from general model
            with self.model_lock:
                general_results = self.general_model(image, verbose=False, iou=0.5, conf=0.8)

            items = []
            for detection in general_results[0].boxes.data:
                # Unpack detection
                x1, y1, x2, y2, confidence, class_id = detection.tolist()

                # Get data of top 5 predictions
                probs = general_results[0].boxes.conf
                classes = general_results[0].boxes.cls
                
                # Sort by confidence and get top 5 class names
                sorted_indices = torch.argsort(probs, descending=True)[:5]
                top5_predictions = [
                    general_results[0].names[int(classes[idx])] for idx in sorted_indices
                ]

                # Get item name from class ID
                item_name = general_results[0].names[int(class_id)]
                
                # Crop image for additional processing
                item_image = image.crop([x1, y1, x2, y2])
                
                # Basic item info
                item_info = CartItem(
                    item = item_name,
                    confidence = confidence,
                    top5_predictions = top5_predictions,
                    bbox = [x1, y1, x2, y2]
                )

                # Process sandwich if detected
                if item_name == 'SANDWICH':
                    result = self.process_sandwich(item_image)

                    if result:
                        # Update item info with sandwich data
                        top5_sandwiches, top_confidence = result
                        item_info = item_info.model_copy(update={
                            'sandwich_item': top5_sandwiches[0][0],
                            'sandwich_confidence': top_confidence,
                            'item': top5_sandwiches[0][0],
                            'top5_predictions': top5_sandwiches
                        })

                # Process data matrix
                #data_matrix_result = self.process_data_matrix(item_image)
                data_matrix_result = None
                if data_matrix_result:
                    matrix_item, matrix_code = data_matrix_result
                    if matrix_item:
                        # Add data matrix item to top 5 predictions
                        current_predictions = item_info.top5_predictions

                        # If matrix_item is already in top5_predictions, move it to the front
                        if matrix_item in current_predictions:
                            top5_predictions = [matrix_item] + [pred for pred in current_predictions if pred != matrix_item]
                        # Otherwise, add it to the front and remove the last item
                        else:
                            top5_predictions = [matrix_item] + current_predictions[:-1]
                        
                        # Update item info with data matrix data
                        item_info = item_info.model_copy(update={
                            'top5_predictions': top5_predictions,
                            'item': matrix_item,
                            'data_matrix_item': matrix_item,
                            'data_matrix': matrix_code
                        })

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
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Create singleton instance
detect = Detect()