import json
import os
import torch
from ultralytics import YOLO
from PIL import Image

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = './dataset'
JSON_PATH = os.path.join(CUR_DIR, 'inconsistent_annotations.json')

def verify_dataset(send_verification_status_route):
    send_verification_status_route('STARTED')

    # Remove exiting verified annotations
    if os.path.exists(JSON_PATH):
        os.remove(JSON_PATH)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model_path = os.path.join(CUR_DIR, '..', 'detection_model', 'models', 'general_model.pt') if os.path.join(CUR_DIR, '..', 'detection_model', 'models', 'general_model.pt') else 'yolo11m.pt'
    MODEL = YOLO(model_path).to(device)

    # Verify annotations
    inconsistent_annotations = []
    splits = ['train', 'valid', 'test']
    for split in splits:
        image_split = os.path.join(DATASET_DIR, split, 'images')
        label_split = os.path.join(DATASET_DIR, split, 'labels')
        for img_name in os.listdir(image_split):
            img_path = os.path.join(image_split, img_name)
            label_path = os.path.join(label_split, os.path.splitext(img_name)[0] + '.txt')

            # Data structure to store inconsistencies
            inconsistencies = {
                'image_path': f'/images/dataset/{split}/images/{os.path.basename(img_path)}',
                'inconsistencies': [],
                'dataset_inconsistency_index': set(),
                'verified_inconsistency_index': set(),
                'verified_annotations': []
            }

            existing_labels = []
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        existing_labels.append((class_id, (x_center, y_center, width, height)))
            except Exception as e:
                print(f'Error reading label file {label_path}: {e}')
                inconsistencies['inconsistencies'].append('ERROR WITH LABELS')
                continue

            annotations = []
            try:
                results = MODEL(img_path, verbose=False)
                for detection in results[0].boxes.data:
                    x1, y1, x2, y2, confidence, class_id = detection.tolist()
                    bbox = bbox_to_yolo([x1, y1, x2, y2], img_path)
                    annotations.append((class_id, bbox))
            except Exception as e:
                print(f'Error verifying image {img_path}: {e}')
                inconsistencies['inconsistencies'].append('ERROR WITH VERIFICATION')
                continue
            
            missing_indexes = check_for_missing_annotations(existing_labels, annotations)
            if missing_indexes:
                inconsistencies['inconsistencies'].append('MISSING ANNOTATIONS')
                inconsistencies['dataset_inconsistency_index'].update(missing_indexes)

            extra_indexes = check_for_extra_annotations(existing_labels, annotations)
            if extra_indexes:
                inconsistencies['inconsistencies'].append('EXTRA ANNOTATIONS')
                inconsistencies['verified_inconsistency_index'].update(extra_indexes)

            dataset_inconsistent_indexes, verified_inconsistent_indexes = check_for_inconsistent_sizes(existing_labels, annotations)
            if dataset_inconsistent_indexes or verified_inconsistent_indexes:
                inconsistencies['inconsistencies'].append('INCONSISTENT SIZES')
                inconsistencies['dataset_inconsistency_index'].update(dataset_inconsistent_indexes)
                inconsistencies['verified_inconsistency_index'].update(verified_inconsistent_indexes)
            
            if inconsistencies['inconsistencies']:
                verified_annotations = []
                for annotation in annotations:
                    x, y, w, h = annotation[1]
                    verified_annotations.append({
                        'class_id': annotation[0],
                        'bbox': [x, y, w, h]
                    })
                inconsistencies['verified_annotations'] = verified_annotations
                inconsistencies['dataset_inconsistency_index'] = list(inconsistencies['dataset_inconsistency_index'])
                inconsistencies['verified_inconsistency_index'] = list(inconsistencies['verified_inconsistency_index'])
                inconsistent_annotations.append(inconsistencies)
        
    # Save inconsistent annotations
    with open(JSON_PATH, 'w') as f:
        json.dump(inconsistent_annotations, f, indent=4)

    send_verification_status_route('DONE')

def bbox_to_yolo(bbox, image_path):
    image = Image.open(image_path)
    image_width = image.width
    image_height = image.height

    x_center = (bbox[0] + bbox[2]) / (2 * image_width)
    y_center = (bbox[1] + bbox[3]) / (2 * image_height)
    bbox_width = (bbox[2] - bbox[0]) / image_width
    bbox_height = (bbox[3] - bbox[1]) / image_height

    return x_center, y_center, bbox_width, bbox_height

def check_for_missing_annotations(existing_labels, annotations):
    missing_indexes = []
    for idx, (label, _) in enumerate(existing_labels):
        if label not in [class_id for class_id, _ in annotations]:
            missing_indexes.append(idx)
    return missing_indexes

def check_for_extra_annotations(existing_labels, annotations):
    extra_indexes = []
    for idx, (class_id, _) in enumerate(annotations):
        if class_id not in [label for label, _ in existing_labels]:
            extra_indexes.append(idx)
    return extra_indexes

def check_for_inconsistent_sizes(existing_labels, annotations, iou_threshold=0.5, size_threshold=0.3):
    dataset_inconsistent_indexes = []
    verified_inconsistent_indexes = []

    for ex_idx, (ex_class, (ex_x, ex_y, ex_w, ex_h)) in enumerate(existing_labels):
        # Flag to track if a consistent annotation is found
        consistent_found = False
        ann_idx = None

        for ann_idx, (ann_class, (ann_x, ann_y, ann_w, ann_h)) in enumerate(annotations):
            # Check if classes match
            if ex_class == ann_class:
                # Compute IoU
                x1 = max(ex_x - ex_w/2, ann_x - ann_w/2)
                y1 = max(ex_y - ex_h/2, ann_y - ann_h/2)
                x2 = min(ex_x + ex_w/2, ann_x + ann_w/2)
                y2 = min(ex_y + ex_h/2, ann_y + ann_h/2)
                
                # Intersection area
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                
                # Union area
                union = ex_w * ex_h + ann_w * ann_h - intersection
                
                # IoU calculation
                iou = intersection / union if union > 0 else 0
                
                # Size difference checks
                width_diff = abs(ex_w - ann_w) / max(ex_w, ann_w)
                height_diff = abs(ex_h - ann_h) / max(ex_h, ann_h)
                
                # If IoU is high and size differences are within threshold
                if (iou >= iou_threshold and 
                    (width_diff <= size_threshold and height_diff <= size_threshold)):
                    consistent_found = True
                    break
        
        # If no consistent annotation found for this label
        if not consistent_found:
            dataset_inconsistent_indexes.append(ex_idx)
            verified_inconsistent_indexes.append(ann_idx)
    
    return dataset_inconsistent_indexes, verified_inconsistent_indexes
