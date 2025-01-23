import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pylibdmtx.pylibdmtx import decode
from pyzbar import pyzbar
import zxingcpp
import time
import statistics
from typing import List, Tuple
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict


# Load YOLO models using Ultralytics
DM_Model = YOLO('YOLOmodels/DMbarcode.pt')
General_Model = YOLO('YOLOmodels/super_general_v2.pt')
Sandwich_Model = YOLO('YOLOmodels/sandwich_classifier.pt')



#New Models
Super_General_Model = YOLO('YOLOmodels/super_general.pt')
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



# DM_Model.to('cuda')
# General_Model.to('cuda')
# Sandwich_Model.to('cuda')

# Define a Detection data class to store detection information
class Detection:
    def __init__(self, top5, bbox, dm_code=None):
        self.top5 = top5
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.dm_code = dm_code  # Decoded Data Matrix code if present

def decode_datamatrix(roi):
    if roi.size != 0:
            # Resize the ROI if larger than 200x200 pixels
            roi_height, roi_width = roi.shape[:2]
            max_dimension = max(roi_width, roi_height)
            DMroi = roi
            if max_dimension > 100:
                scaling_factor = 100 / max_dimension
                new_width = int(roi_width * scaling_factor)
                new_height = int(roi_height * scaling_factor)
                DMroi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)

            decoded_info = decode(DMroi, max_count=1, shape=2, min_edge=30, threshold=50)
            #print("Decoded Info:", decoded_info)    
    if decoded_info:
        dm_code = decoded_info[0].data.decode('utf-8')
        return dm_code
    else:
        resized = roi
        resized = cv2.resize(roi, (640  ,640 ), interpolation = cv2.INTER_CUBIC) 
        resized = cv2.GaussianBlur(resized, (5, 5), 0)
        kernel = np.array([[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]])
        resized = cv2.filter2D(resized, -1, kernel)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sharpening_kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
        #resized = cv2.filter2D(resized, -1, sharpening_kernel)
        
        # resized = cv2.adaptiveThreshold(
        #     resized,
        #     255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY,
        #     11,
        #     2
        # )
        
        #cv2.imshow('BRUH', resized)
        zxingcpp_results = zxingcpp.read_barcodes(resized)
        if zxingcpp_results == []:
            print("Pxzing-cpp Results: None")
        else:
            #print("Pxzing-cpp Results:", zxingcpp_results[0].text)
            return zxingcpp_results[0].text
            
        pyzbar_results = pyzbar.decode(resized, symbols=[pyzbar.ZBarSymbol.EAN13])
        #print("Pxzing Results:", zxingreader.decode('BRUH.png'))
        #print("Pyzbar Results:", pyzbar_results)
        if pyzbar_results:
            dm_code = pyzbar_results[0].data.decode('utf-8')
            return dm_code
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
    
    
# def classify_roi(label, roi):
#     """
#     Classify the ROI using the specified label's model. If the label is not found,
#     aggregate predictions from all models and return combined top5 results.
#     """
#     model_mapping = {
#         "BAR": BAR_Model,
#         "BOTTLE": BOTTLE_Model,
#         "CAN": CAN_Model,
#         "CHIPS": CHIPS_Model,
#         "CUP": CUP_Model,
#         "OTHER": OTHER_Model,
#         "SANDWICH": SANDWICH1_Model,
#         "SPOONFUL": SPOONFUL_Model,
#     }

#     model_func = model_mapping.get(label)
    
#     if model_func:
#         # Use the specific model for the label
#         result = model_func(roi)
#         names = result[0].names
#         top5 = result[0].probs.top5
#         top5conf = result[0].probs.top5conf
#         top5array = []
#         for i in range(len(top5)):
#             top5array.append((names[top5[i]], top5conf[i].item()))
#         return top5array
#     else:
#         # Aggregate predictions from ALL models
#         aggregated_confidences = defaultdict(float)
        
#         # Run every model in the mapping
#         for current_model in model_mapping.values():
#             result = current_model(roi)
#             names = result[0].names
#             top5_indices = result[0].probs.top5
#             top5_conf = result[0].probs.top5conf
            
#             # Collect confidences for all labels
#             for i in range(len(top5_indices)):
#                 label_name = names[top5_indices[i]]
#                 confidence = top5_conf[i].item()
#                 aggregated_confidences[label_name] += confidence
        
#         # Sort aggregated results and take top5
#         sorted_results = sorted(
#             [(conf, label) for label, conf in aggregated_confidences.items()],
#             key=lambda x: x[0],
#             reverse=True
#         )[:5]  # Keep only top5 entries
        
#         # Format as (label, confidence) tuples
#         top5_combined = [(label, conf) for conf, label in sorted_results]
        
#         return top5_combined if top5_combined else None   
    


def detect(frame):
    detections = []
    original_height, original_width = frame.shape[:2]

    # Resize frame to 640x640 by squashing (without maintaining aspect ratio)
    frame_resized = cv2.resize(frame, (640, 640))

    # Calculate scaling factors
    scale_x = original_width / 640
    scale_y = original_height / 640

    # Run detection on the resized frame
    general_results = General_Model(frame_resized, stream=True, conf = 0.6)

    for result in general_results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Map bounding boxes back to original frame coordinates
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            label = General_Model.names[class_id]

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_width - 1, x2)
            y2 = min(original_height - 1, y2)

            # Extract the ROI from the original high-resolution frame
            roi = frame[y1:y2, x1:x2]
            

            classification_results = classify_roi(label, roi)

            


            dm_code = None  # Initialize dm_code as None

            
            if classification_results[0][1] > 0.1:
                # For other items, check for Data Matrix codes using DM_Model
                dm_results = DM_Model(roi, stream=True, verbose=False)
                for dm_result in dm_results:
                    for dm_box in dm_result.boxes:
                        dm_confidence = dm_box.conf[0].item()
                        # Get coordinates of Data Matrix code detection relative to the ROI
                        dx1, dy1, dx2, dy2 = map(int, dm_box.xyxy[0].cpu().numpy())
                        # Ensure coordinates are within ROI bounds
                        dx1 = max(0, dx1)
                        dy1 = max(0, dy1)
                        dx2 = min(roi.shape[1] - 1, dx2)
                        dy2 = min(roi.shape[0] - 1, dy2)
                        # Extract the Data Matrix code region from ROI
                        dm_roi = roi[int(dy1*0.95):int(dy2*1.05), int(dx1*0.95):int(dx2*1.05)]
                        # Decode the Data Matrix code
                        #print("Item:", DM_Model.names[dm_box.cls[0].item()])
                        dm_code = decode_datamatrix(dm_roi)
                        if dm_code:
                            break  # Found a Data Matrix code
                    if dm_code:
                        break  # Exit outer loop if DM code is found

            # Create a Detection object
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



# if __name__ == "__main__":
#     # Initialize video capture with a video file
#     video_path = "Media/sushi.mp4"  # Replace with your video file path
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         exit()

#     while True:
#         ret, frame = cap.read()
#         if not ret:  # Break the loop if the video ends
#             break

#         detections = detect(frame)
#         frame = draw_detections(frame, detections)

#         cv2.imshow('Frame', frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    

    
    
    
if __name__ == "__main__":
    # Initialize video capture (ensure your webcam supports 1080p)
    cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
                detections = detect(frame)
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
            confidence_threshold=0.5,  # Adjust as needed
            )

            print("Final Predictions:", final_predictions)
        
        prev_frame = frame
        # print the size of the location_groups
        # You can now customize the confidence threshold
        
        # take detection
        # we will have a list of locations
        # first look at a bounding box and find the centre of the bounding box
        # If the centre of the bounding box is close enough to a location in a list, then you add it to that list

        
        # After 2 seconds of filling up the list we will analyse the list
        
        # Keep track of the number of frames

        # for each location, we need to find the most common item in the list
        # if there is a datamatrix/barcode deetected, then we will use that as the item, we will also set the confidence to 1
        # add up all the confidence values for each item and pick the item with the highest confidence sum

        #frame = draw_detections(frame, detections)
        #resized = frame
        #resized = cv2.resize(frame, (1280 ,720 ), interpolation = cv2.INTER_AREA) 
        #cv2.imshow('Frame', resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
