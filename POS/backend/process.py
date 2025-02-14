import time
import statistics
from collections import defaultdict
from typing import List, Tuple, Optional
from ultralytics import YOLO
import os
import cv2
import pandas as pd
import numpy as np
from skimage import img_as_float
import time
from pylibdmtx.pylibdmtx import decode
from pyzbar import pyzbar
import zxingcpp

DEBUG = False

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIERS_DIR = os.path.join(CUR_DIR, 'YOLOmodels/classifiers')

# Data matrix model
DM_MODEL = YOLO(os.path.join(CUR_DIR, 'YOLOmodels', 'DMbarcode.pt'))

# General object detection model
GENERAL_MODEL = YOLO(os.path.join(CUR_DIR, 'YOLOmodels', 'general_model.pt'))

# Create a map of all the classifiers
# NOTE: The name of the classifier model file must match the name of the class in the general model
MODELS_MAP = {}
for model_file in os.listdir(CLASSIFIERS_DIR):
    if model_file.endswith('.pt'):
        # Get model name and path
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(CLASSIFIERS_DIR, model_file)

        # Put model into map
        MODELS_MAP[model_name] = YOLO(model_path)

# Product list
class ProductLookup:
    def __init__(self, product_list_path=os.path.join(CUR_DIR, 'product_list.xlsx')):
        self.product_list = pd.read_excel(product_list_path).astype(str)
    
    def find_product_name(self, barcode):
        product = self.product_list[self.product_list['barcode'] == barcode]
        if not product.empty:
            return product['DisplayName'].values[0]
        else:
            return 'UNKNOWN'

PRODUCT_LIST = ProductLookup(os.path.join(CUR_DIR, 'product_list.xlsx'))

print("\n\n# ----- Initialisation Complete ----- #\n\n")

class Detection:
    """
    Simple container for detection results.

    Attributes:
        top5 (List[Tuple[str, float]]): 
            Top-5 classifications or predictions in the format (label, confidence).
        bbox (Tuple[int, int, int, int]): 
            Bounding box represented as (x1, y1, x2, y2).
        dm_code (Optional[str]): 
            Decoded Data Matrix code (if available).
    """
    def __init__(
        self,
        top5: List[Tuple[str, float]],
        bbox: Tuple[int, int, int, int],
        dm_code: Optional[str] = None
    ):
        self.top5 = top5
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.dm_code = dm_code  # Decoded Data Matrix code if present

def decode_datamatrix(roi: np.ndarray) -> Optional[str]:
    """
    Attempt to decode a Data Matrix or barcode from a given region of interest (ROI).

    Steps:
        1. If ROI is non-empty:
            a. Optionally resize ROI if larger than 100x100.
            b. Attempt to decode using pylibdmtx.
        2. If that fails, apply a series of filters and attempt to decode using:
            a. zxingcpp
            b. pyzbar (for EAN13)
    
    Args:
        roi (np.ndarray): The image ROI from which to decode the Data Matrix/Barcode.

    Returns:
        Optional[str]: The decoded string if successful, otherwise None.
    """

    # 1. Early exit if ROI is empty
    if roi.size == 0:
        return None

    # 2. Resize ROI if the largest dimension > 100 to improve decoding reliability
    roi_height, roi_width = roi.shape[:2]
    max_dimension = max(roi_width, roi_height)
    dm_roi = roi  # Keep an unmodified reference in case we need it later

    if max_dimension > 100:
        scaling_factor = 100 / max_dimension
        new_width = int(roi_width * scaling_factor)
        new_height = int(roi_height * scaling_factor)
        dm_roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 3. Attempt to decode using pylibdmtx (Data Matrix focus)
    decoded_info = decode(dm_roi, max_count=1, shape=2, min_edge=30, threshold=50)
    if decoded_info:
        # If decoding is successful, return the decoded string
        return decoded_info[0].data.decode('utf-8')

    # 4. If pylibdmtx fails, proceed with additional processing for other barcode types (zxingcpp and pyzbar)
    # 4a. Resize to a consistent dimension to improve decoding
    resized = cv2.resize(roi, (640, 640), interpolation=cv2.INTER_CUBIC)

    # 4b. Apply a Gaussian blur to reduce noise
    resized = cv2.GaussianBlur(resized, (5, 5), 0)

    # 4c. Sharpen the image (mild filter)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    resized = cv2.filter2D(resized, -1, sharpen_kernel)

    # 4d. Convert to grayscale for easier barcode detection
    resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 5. Attempt decode with zxingcpp
    zxingcpp_results = zxingcpp.read_barcodes(resized_gray)
    if zxingcpp_results:
        return zxingcpp_results[0].text

    # 6. Attempt decode with pyzbar (e.g., EAN13)
    pyzbar_results = pyzbar.decode(resized_gray, symbols=[pyzbar.ZBarSymbol.EAN13])
    if pyzbar_results:
        return pyzbar_results[0].data.decode('utf-8')

    # 7. If all methods fail, return None
    return None

def is_frame_different(frame1, frame2, threshold=0.01):
    '''
    Calculates the difference between two frames and returns whether the difference is greater than the threshold
    Threshold is the normalised score between 0 and 1 (0 = no difference, 1 = completely different)
    Default threshold of 0.01 was observed to be a good value for differentiating between noise in static frames and actual changes
    '''
    frame1 = img_as_float(frame1)
    frame2 = img_as_float(frame2)

    # Different formulas can be used - mae is less computationally expensive
    mae = np.mean(np.abs(frame1 - frame2))
    if DEBUG:
        print("MAE: ", mae)
    return mae > threshold

def detect_objects_in_frame(frame: np.ndarray) -> List[Detection]:
    '''
    Main function to detect objects in a given frame, classify each object, and optionally detect and decode Data Matrix codes.

    Args:
        frame (np.ndarray): The original image/frame in BGR format.

    Returns:
        detections (List[Detection]): List of Detection objects containing:
            - top5: classification results
            - bbox: bounding box coordinates (x1, y1, x2, y2)
            - dm_code: Decoded DM code if detected
    '''
    detections = []
    general_results = GENERAL_MODEL(frame, stream=True, conf=0.6, verbose=DEBUG)

    for result in general_results:
        for box in result.boxes:
            # Extract YOLO output
            class_id = int(box.cls[0].item())
            label = GENERAL_MODEL.names[class_id]
            confidence = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Get region of interest of the item
            roi = frame[y1:y2, x1:x2]

            # Classify the ROI to get the specific item name and data matrix code
            classification_results, dm_code = identify_item(label, confidence, roi)

            # Add the detection to the list
            detection = Detection(classification_results, (x1, y1, x2, y2), dm_code)
            detections.append(detection)

    return detections

def identify_item(label: str, confidence: float, roi: np.ndarray, classification_threshold: float = 0.9) -> (List, str):
    '''
    Classify the ROI, and if confidence is above threshold, attempt DM detection.
    
    Args:
        label (str): Label from the general detection model.
        roi (np.ndarray): The Region of Interest (object bounding box in original frame).
        classification_threshold (float): Threshold to decide if we should check 
                                          for a Data Matrix code.

    Returns:
        (classification_results, dm_code):
            classification_results (List): The result from classify_roi(label, roi).
            dm_code (str or None): The detected DM code, if any.
    '''
    classification_results = classify_roi(label, confidence, roi)

    LABEL_WITH_DM = ['TF-BROWN', 'TF-BLUE', 'SANDWICH', 'BURRITO']
    dm_code = None
    if label in LABEL_WITH_DM or classification_results[0][1] < classification_threshold:
        dm_code = detect_datamatrix_in_roi(roi)
    
    return classification_results, dm_code

def classify_roi(label, confidence, roi):
    '''
    Classify the ROI based on the label by calling the appropriate model function.
    Returns the model result (e.g., probability, predicted class, etc.).
    '''
    model_func = MODELS_MAP.get(label)

    # If there is a specific model for the label, use it
    if model_func:
        result = model_func(roi, verbose=DEBUG)
        names = result[0].names
        top5 = result[0].probs.top5
        top5conf = result[0].probs.top5conf
        top5array = []
        for i in range(len(top5)):
            top5array.append((names[top5[i]], top5conf[i].item()))
        
        return top5array
    # Otherwise, return the general model result        
    else:
        return [(label, confidence)]

def detect_datamatrix_in_roi(roi: np.ndarray) -> str:
    """
    Detect and decode a Data Matrix code within the given ROI using DM_Model.
    
    Args:
        roi (np.ndarray): Region of interest from the original image 
                          (where an object has been detected).

    Returns:
        dm_code (str): Decoded Data Matrix code, or None if not found.
    """
    dm_code = None
    dm_results = DM_MODEL(roi, stream=True, verbose=DEBUG)
    for dm_result in dm_results:
        for dm_box in dm_result.boxes:
            # Coordinates are relative to the ROI
            dx1, dy1, dx2, dy2 = map(int, dm_box.xyxy[0].cpu().numpy())

            # Clamp to ROI bounds
            dx1 = max(0, dx1)
            dy1 = max(0, dy1)
            dx2 = min(roi.shape[1] - 1, dx2)
            dy2 = min(roi.shape[0] - 1, dy2)

            # Slightly expand the bounding box for decoding
            expanded_roi = roi[int(dy1 * 0.95): int(dy2 * 1.05), 
                               int(dx1 * 0.95): int(dx2 * 1.05)]

            # Attempt to decode the Data Matrix
            dm_code = decode_datamatrix(expanded_roi)
            if dm_code:
                break  # Found a Data Matrix code

        if dm_code:
            break  # Exit outer loop if DM code is found

    return dm_code

def track_detections(location_groups: List[List[Detection]], current_detections: List[Detection], proximity_threshold: int = 200) -> List[List[Detection]]:
    """
    Continuously update location groups with new detections.
    
    Args:
        location_groups: Existing groups of detections
        current_detections: New detections from the current frame
        proximity_threshold: Maximum distance between detection centers to be considered the same group
    
    Returns:
        Updated list of detection groups
    """
    for detection in current_detections:
        # Calculate center of current detection's bounding box
        x1, y1, x2, y2 = detection.bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Flag to track if detection was added to an existing group
        detection_added = False
        
        # Check existing location groups
        for group in location_groups:
            # Calculate group's center (average of all detection centers in the group)
            group_center_x = int(statistics.mean((det.bbox[0] + det.bbox[2]) // 2 for det in group))
            group_center_y = int(statistics.mean((det.bbox[1] + det.bbox[3]) // 2 for det in group))
            
            # Check if detection is close to group's center
            if (abs(center_x - group_center_x) < proximity_threshold and 
                abs(center_y - group_center_y) < proximity_threshold):
                
                # Prioritize adding detections with Data Matrix/Barcode
                if detection.dm_code is not None:
                    # If group doesn't have a DM/Barcode, add at the beginning
                    if not any(det.dm_code for det in group):
                        group.insert(0, detection)
                    else:
                        group.append(detection)
                else:
                    # Add non-DM/Barcode detection to the end of the group
                    group.append(detection)
                
                detection_added = True
                break
        
        # If detection wasn't added to any existing group, create a new group
        if not detection_added:
            location_groups.append([detection])
    
    return location_groups

def process_predictions(location_groups: List[List[Detection]], 
                        frames_counted: int, 
                        confidence_threshold: float = 0.3
                        ) -> List[List[Tuple[float, str]]]:
    """
    Process location groups to determine predictions using top5 confidences.
    
    Args:
        location_groups: List of detection groups
        frames_counted: Number of frames used for tracking
        confidence_threshold: Minimum confidence to keep an item in predictions
    
    Returns:
        List of predictions for each location, sorted by confidence in descending order
    """
    location_predictions = []
    
    for group in location_groups:
        item_stats = {}
        
        for detection in group:
            # Handle detections with Data Matrix code
            if detection.dm_code:
                product_name = PRODUCT_LIST.find_product_name(detection.dm_code)
                if product_name != "UNKNOWN":
                    if product_name not in item_stats:
                        item_stats[product_name] = {
                            'count': 1,
                            'total_confidence': 1.5
                        }
                    else:
                        item_stats[product_name]['count'] += 1
                        item_stats[product_name]['total_confidence'] += 1.5
                continue  # Skip top5 processing if DM code is valid
            
            # Handle detections without Data Matrix code (process all top5 entries)
            for label, confidence in detection.top5:
                if label not in item_stats:
                    item_stats[label] = {
                        'count': 1,
                        'total_confidence': confidence
                    }
                else:
                    item_stats[label]['count'] += 1
                    item_stats[label]['total_confidence'] += confidence
        
        # Calculate final confidences
        location_confidences = []
        for item, stats in item_stats.items():
            final_confidence = stats['total_confidence'] / frames_counted
            location_confidences.append((final_confidence, item))
        
        # Sort and filter
        location_confidences.sort(reverse=True, key=lambda x: x[0])

        if float(location_confidences[0][0]) > confidence_threshold:
            location_predictions.append([location_confidences[0]])
    
    return location_predictions

if __name__ == "__main__":
    # Initialize video capture (ensure your webcam supports 1080p)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

    ret, frame = cap.read()
    prev_frame = frame
  
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if frame is different from previous frame
        if is_frame_different(prev_frame, frame):
            print("Frame Difference Detected, Waiting for Frame Stability...")

            # Wait until frame is stable
            while is_frame_different(frame, prev_frame, threshold=0.03):
                prev_frame = frame
                ret, frame = cap.read()
                if not ret:
                    break
            
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    break
            
            print("Frame is Stable, Processing...")
            location_groups = []
            frames_counted = 0

            start_time = time.time()
            
            # Process the next 5 frames for redundancy
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detect_objects_in_frame(frame)
                location_groups = track_detections(location_groups, detections)
                frames_counted += 1

            final_predictions = process_predictions(location_groups, frames_counted, confidence_threshold=0.5)
            print("Total Detection Time:", time.time() - start_time)
            print("Final Predictions:", final_predictions, '\n\n')
        
        prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()
