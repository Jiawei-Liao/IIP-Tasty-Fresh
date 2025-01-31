# --- Standard Library Imports ---
import time
import statistics
from collections import defaultdict
from typing import List, Tuple, Optional

# --- Third-Party Library Imports ---
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pylibdmtx.pylibdmtx import decode
from pyzbar import pyzbar
import zxingcpp
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# --- YOLO Model Loading ---

# Data matrix model
DM_Model = YOLO('YOLOmodels/DMbarcode.pt')

# General object detection model
General_Model = YOLO('YOLOmodels/general_model.pt')

# classifier models
BAR_Model = YOLO('YOLOmodels/classifiers/BAR.pt')
BOTTLE_Model = YOLO('YOLOmodels/classifiers/BOTTLE.pt')
CAN_Model = YOLO('YOLOmodels/classifiers/CAN.pt')
CHIPS_Model = YOLO('YOLOmodels/classifiers/CHIPS.pt')
CUP_Model = YOLO('YOLOmodels/classifiers/CUP.pt')
OTHER_Model = YOLO('YOLOmodels/classifiers/OTHER.pt')
SANDWICH1_Model = YOLO('YOLOmodels/classifiers/SANDWICH.pt')
SPOONFUL_Model = YOLO('YOLOmodels/classifiers/SPOONFUL.pt')
TF_BLUE_Model = YOLO('YOLOmodels/classifiers/TF-BLUE.pt')
TF_BROWN_Model = YOLO('YOLOmodels/classifiers/TF-BROWN.pt')

# Uncomment if you want to run these models on GPU:
# DM_Model.to('cuda')
# General_Model.to('cuda')
# Sandwich_Model.to('cuda')

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

    # 4. If pylibdmtx fails, proceed with additional processing for other barcode types
    #    (zxingcpp and pyzbar)

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

    # Optionally apply a more intense sharpening kernel:
    # intense_sharpen = np.array([[-1, -1, -1],
    #                             [-1,  9, -1],
    #                             [-1, -1, -1]])
    # resized_gray = cv2.filter2D(resized_gray, -1, intense_sharpen)

    # Optionally apply adaptive thresholding:
    # resized_gray = cv2.adaptiveThreshold(
    #     resized_gray,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     11,
    #     2
    # )

    # 5. Attempt decode with zxingcpp
    zxingcpp_results = zxingcpp.read_barcodes(resized_gray)
    if zxingcpp_results:
        return zxingcpp_results[0].text
    else:
        print("zxingcpp Results: None")

    # 6. Attempt decode with pyzbar (e.g., EAN13)
    pyzbar_results = pyzbar.decode(resized_gray, symbols=[pyzbar.ZBarSymbol.EAN13])
    if pyzbar_results:
        return pyzbar_results[0].data.decode('utf-8')

    # 7. If all methods fail, return None
    return None

def classify_roi(label, roi):
    """
    Classify the ROI based on the label by calling the appropriate model function.
    Returns the model result (e.g., probability, predicted class, etc.).
    """
    # Define a dictionary that maps each label to its corresponding model function
        
    model_mapping = {
        "BAR": BAR_Model,
        "BOTTLE": BOTTLE_Model,
        "CAN": CAN_Model,
        "CHIPS": CHIPS_Model,
        "CUP": CUP_Model,
        #"OTHER": OTHER_Model,
        "SANDWICH": SANDWICH1_Model,
        "SPOONFUL": SPOONFUL_Model,
        "TF-BLUE": TF_BLUE_Model,
        "TF-BROWN": TF_BROWN_Model
    }

    # Get the model function based on the label
    model_func = model_mapping.get(label)
    
    if model_func:
        result = model_func(roi)
        names = result[0].names
        top5 = result[0].probs.top5
        top5conf = result[0].probs.top5conf
        top5array = []
        for i in range(len(top5)):
            top5array.append((names[top5[i]], top5conf[i].item()))
        
        return top5array
        
    else:
        # Fallback if the label is not in the dictionary
        print(f"No specific model found for label '{label}'. Returning None.")
        return None

def preprocess_frame(
    frame: np.ndarray, 
    target_size: int = 640
) -> (np.ndarray, float, float):
    """
    Resize the input frame to a square of target_size x target_size (squash), 
    and compute scaling factors to map detections back to the original size.
    
    Args:
        frame (np.ndarray): Original image.
        target_size (int): Desired size to reshape the image (width and height).

    Returns:
        resized_frame (np.ndarray): Frame resized to (target_size x target_size).
        scale_x (float): Scaling factor to restore width to original size.
        scale_y (float): Scaling factor to restore height to original size.
    """
    original_height, original_width = frame.shape[:2]

    # Resize frame (squash) to target_size x target_size
    resized_frame = cv2.resize(frame, (target_size, target_size))

    # Calculate scaling factors
    scale_x = original_width / target_size
    scale_y = original_height / target_size

    return resized_frame, scale_x, scale_y

def run_general_detection(
    resized_frame: np.ndarray, 
    conf_threshold: float = 0.6
):
    """
    Run the General_Model on the resized frame to get detection results.
    
    Args:
        resized_frame (np.ndarray): The frame resized to 640x640.
        conf_threshold (float): Confidence threshold for YOLO detection.

    Returns:
        generator of results (from YOLO.stream)
    """
    return General_Model(resized_frame, stream=True, conf=conf_threshold)

def map_bounding_box(
    box_coords: np.ndarray, 
    scale_x: float, 
    scale_y: float
) -> (int, int, int, int):
    """
    Map bounding box coordinates from the resized frame back to the original frame.
    
    Args:
        box_coords (np.ndarray): Bounding box [x1, y1, x2, y2] in resized scale.
        scale_x (float): Scaling factor for width.
        scale_y (float): Scaling factor for height.

    Returns:
        (x1, y1, x2, y2) mapped to the original image scale.
    """
    x1, y1, x2, y2 = box_coords
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    return x1, y1, x2, y2

def clamp_bounding_box(
    x1: int, y1: int, x2: int, y2: int, 
    width: int, height: int
) -> (int, int, int, int):
    """
    Ensure bounding box coordinates stay within image bounds.
    
    Args:
        x1, y1, x2, y2 (int): Original bounding box coordinates.
        width, height (int): Dimensions of the original frame.
        
    Returns:
        (x1, y1, x2, y2) clamped within [0, width] and [0, height].
    """
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    return x1, y1, x2, y2

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
    dm_results = DM_Model(roi, stream=True, verbose=False)
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

def classify_and_maybe_decode_dm(
    label: str, 
    roi: np.ndarray, 
    classification_threshold: float = 0.1
) -> (List, str):
    """
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
    """
    classification_results = classify_roi(label, roi)  # Provided by your code
    dm_code = None

    # classification_results[0] => (class_label, confidence)
    if classification_results[0][1] > classification_threshold:
        dm_code = detect_datamatrix_in_roi(roi)

    return classification_results, dm_code

def detect_objects_in_frame(frame: np.ndarray) -> List[Detection]:
    """
    Main function to detect objects in a given frame, classify each object, 
    and optionally detect and decode Data Matrix codes.

    Args:
        frame (np.ndarray): The original image/frame in BGR format.

    Returns:
        detections (List[Detection]): List of Detection objects containing:
            - top5: classification results
            - bbox: bounding box coordinates (x1, y1, x2, y2)
            - dm_code: Decoded DM code if detected
    """
    detections = []
    original_height, original_width = frame.shape[:2]

    # 1. Preprocess/resize frame
    frame_resized, scale_x, scale_y = preprocess_frame(frame)

    # 2. Run detection on the resized frame
    general_results = run_general_detection(frame_resized, conf_threshold=0.6)

    # 3. Iterate over detection results
    for result in general_results:
        for box in result.boxes:
            # Extract YOLO output
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            x1y1x2y2 = box.xyxy[0].cpu().numpy()

            # 3a. Map bounding box back to original coordinates
            x1, y1, x2, y2 = map_bounding_box(x1y1x2y2, scale_x, scale_y)

            # 3b. Clamp coordinates to image bounds
            x1, y1, x2, y2 = clamp_bounding_box(x1, y1, x2, y2, original_width, original_height)

            # 3c. Extract ROI from the original (high-resolution) frame
            roi = frame[y1:y2, x1:x2]

            # 3d. Classify ROI and optionally detect Data Matrix codes
            label = General_Model.names[class_id]
            classification_results, dm_code = classify_and_maybe_decode_dm(label, roi)

            # 4. Create a Detection object and append to results
            detection = Detection(classification_results, (x1, y1, x2, y2), dm_code)
            detections.append(detection)

    return detections

def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        label = f"{detection.top5[0][0]} {detection.top5[0][1]:.2f}"
        color = (0, 255, 0)  # Green color for bounding boxes

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Put label above the bounding box
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # If a Data Matrix code is present, display it below the bounding box
        if detection.dm_code:
            cv2.putText(
                frame,
                f"DM: {detection.dm_code}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),  # Blue color for DM code
                2
            )
    return frame

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

def create_product_dataframe(excel_path):
    """
    Create and preprocess the product lookup DataFrame.
    
    Args:
        excel_path (str): Path to the Excel file
    
    Returns:
        tuple: (DataFrame, barcode column name, description column name)
    """
    try:
        # Read the Excel file
        # Use dtype=str to ensure barcodes are read as strings and prevent type conversion issues
        df = pd.read_excel(excel_path, dtype=str)
        
        # Normalize column names (remove whitespace, convert to lowercase)
        df.columns = [col.strip().lower().replace(' ', '') for col in df.columns]
        
        # Try multiple potential column names for barcode
        barcode_columns = ['barcode', 'lookupcode', 'barcodenum', 'code']
        
        # Find the first matching barcode column
        barcode_col = None
        for col in barcode_columns:
            if col in df.columns:
                barcode_col = col
                break
        
        if barcode_col is None:
            print("No barcode column found in the Excel file")
            return None, None, None
        
        # Try to find the product description column
        desc_columns = ['productdescription', 'product', 'description', 'name']
        desc_col = None
        for col in desc_columns:
            if col in df.columns:
                desc_col = col
                break
        
        if desc_col is None:
            print("No product description column found")
            return None, None, None
        
        return df, barcode_col, desc_col
    
    except FileNotFoundError:
        print(f"Excel file not found: {excel_path}")
        return None, None, None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, None, None

def lookup_product_by_barcode(df, barcode_col, desc_col, barcode):
    """
    Look up product details by barcode in a preprocessed DataFrame.
    
    Args:
        df (pandas.DataFrame): Preprocessed DataFrame
        barcode_col (str): Name of the barcode column
        desc_col (str): Name of the description column
        barcode (str): Barcode to look up
    
    Returns:
        str: Product name or 'UNKNOWN' if not found
    """
    if df is None or barcode_col is None or desc_col is None:
        return "UNKNOWN"
    
    # Find the row with matching barcode
    matching_rows = df[df[barcode_col] == str(barcode)]
    
    if not matching_rows.empty:
        # Return the first matching product description
        return matching_rows[desc_col].iloc[0]
    
    return "UNKNOWN"

# Example of how to use in your main script
class ProductLookup:
    def __init__(self, excel_path):
        self.df, self.barcode_col, self.desc_col = create_product_dataframe(excel_path)
    
    def find_product_name(self, barcode):
        return lookup_product_by_barcode(self.df, self.barcode_col, self.desc_col, barcode)

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
                product_name = product_lookup.find_product_name(detection.dm_code)
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
        # filtered_confidences = [
        #     (conf, label) for conf, label in location_confidences 
        #     if conf >= confidence_threshold
        # ]
        
        # if not filtered_confidences and location_confidences:
        #     filtered_confidences = [location_confidences[0]]
        
        #location_predictions.append(filtered_confidences)
    
    return location_predictions

def detect_frame_difference(prev_frame, curr_frame, threshold=0.1):
    """
    Compare the current frame and previous frame using SSIM.
    
    Parameters:
    - prev_frame: np.ndarray - The previous frame (grayscale or RGB image).
    - curr_frame: np.ndarray - The current frame (grayscale or RGB image).
    - threshold: float - The SSIM difference threshold. If the SSIM is less than 
                 (1 - threshold), the function returns True (indicating a significant change).
    
    Returns:
    - bool - True if the difference exceeds the threshold, False otherwise.
    """
    # Ensure both frames are the same shape
    prev_frame = cv2.resize(prev_frame, (720 ,480 ), interpolation = cv2.INTER_AREA) 
    curr_frame = cv2.resize(curr_frame, (720 ,480 ), interpolation = cv2.INTER_AREA)

    if prev_frame.shape != curr_frame.shape:
        raise ValueError("Both frames must have the same dimensions.")
    
    # Convert frames to grayscale if they are RGB
    if len(prev_frame.shape) == 3 and prev_frame.shape[2] == 3:
        prev_frame = np.dot(prev_frame[...,:3], [0.2989, 0.5870, 0.1140])
        curr_frame = np.dot(curr_frame[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Ensure the frames are in the proper range (0 to 255 or 0 to 1)
    if prev_frame.max() > 1.0 or curr_frame.max() > 1.0:
        data_range = 255  # Assuming images are in 8-bit (0 to 255)
    else:
        data_range = 1.0  # For normalized images (0 to 1)
    
    # Compute SSIM
    ssim_index, _ = ssim(prev_frame, curr_frame, full=True, data_range=data_range)
    #print("SSIM Index:", ssim_index)
    # Check if the difference exceeds the threshold
    return (1 - ssim_index) > threshold

if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_V4L)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

    product_lookup = ProductLookup('product_list.xlsx')
    ret, frame = cap.read()
    prev_frame = frame
  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # check frames for large variations.
        # if there is a large variation, initiate the detection process

        # while time.elapsed < 2 seconds, keep filling up the list
        # locations = []

        # if time.elapsed > 2 seconds, then we will analyse the list
        cv2.imshow('Frame', frame)
        if detect_frame_difference(prev_frame, frame):
            location_groups = []
            frames_counted = 0
            print("Frame Difference Detected")

            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    break
                start = time.time()
                detections = detect_objects_in_frame(frame)
                location_groups = track_detections(location_groups, detections)
                print("Detection Time:", time.time() - start)
                frame = draw_detections(frame, detections)
                resized = frame
                resized = cv2.resize(frame, (1280 ,720 ), interpolation = cv2.INTER_AREA) 
                cv2.imshow('Frame', resized)
                cv2.waitKey(1)
                frames_counted += 1

            final_predictions = process_predictions(
            location_groups, 
            frames_counted, 
            confidence_threshold=0.5,
            )

            print("Final Predictions:", final_predictions)
        
        prev_frame = frame
        cv2.waitKey(1000)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
