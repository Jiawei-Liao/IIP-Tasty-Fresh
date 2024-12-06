import threading
import os
from datetime import datetime
import shutil
import json
import glob
import torch
from ultralytics import YOLO
from datetime import datetime
from rembg import remove, new_session
from PIL import Image
import random
import re

# Directories
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA_DIR = os.path.join(CUR_DIR, './new_data')
ANNOTATIONS_DIR = os.path.join(CUR_DIR, './annotations')
EXISTING_DATA_DIR = os.path.join(CUR_DIR, './../dataset')
REMBG_CROPPED_IMG_DIR = os.path.join(CUR_DIR, './rembg_cropped_images')
DATASET_DIR = os.path.join(CUR_DIR, './tmp_dataset')

# Tiers
TIERS = {
    'Tier 1': 'tier_1',
    'Tier 2': 'tier_2',
    'Tier 3': 'tier_3'
}

def save_uploaded_images(images, upload_type, class_name, send_annotation_status):
    """Save uploaded images to disk"""
    # Clear existing images
    if os.path.exists(NEW_DATA_DIR):
        shutil.rmtree(NEW_DATA_DIR)
    for tier in TIERS.values():
        os.makedirs(os.path.join(NEW_DATA_DIR, tier), exist_ok=True)
    
    # Save uploaded images
    for image in images:
        metadata = json.loads(image.filename)
        filename = metadata['name']
        tier = 'tier_3' if upload_type == 'existing' else TIERS[metadata['tier']]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]

        save_path = os.path.join(NEW_DATA_DIR, tier, f'{filename}.{timestamp}.png')

        image.save(save_path)
    
    # Annotate images
    threading.Thread(target=annotate_images, args=(upload_type, class_name, send_annotation_status)).start()

# Variables for annotation
NEW_CLASS_ID = None
UPLOAD_TYPE = None
CLASS_NAME = None
MODEL = None
YAML_CONTENT = None
NC = None
NEW_CLASS = False
def annotate_images(upload_type, class_name, send_annotation_status):
    send_annotation_status('STARTED')

    # Setup model and variables
    global NEW_CLASS_ID, UPLOAD_TYPE, CLASS_NAME, MODEL, YAML_CONTENT, NC, NEW_CLASS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(CUR_DIR, 'general_model.pt')
    MODEL = YOLO(model_path).to(device)
    UPLOAD_TYPE = upload_type
    CLASS_NAME = class_name
    YAML_CONTENT = None
    if class_name in MODEL.names:
        NEW_CLASS = False
        NEW_CLASS_ID = list(MODEL.names.values()).index(class_name)
        NC = len(MODEL.names)
    else:
        NEW_CLASS = True
        NEW_CLASS_ID = len(MODEL.names)
        NC = len(MODEL.names) + 1
    
    # Reset directories
    # Finished annotations
    annotation_path = os.path.join(CUR_DIR, 'annotations')
    if os.path.exists(annotation_path):
        shutil.rmtree(annotation_path)
    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(annotation_path, subdir), exist_ok=True)

    # Removed background images
    if os.path.exists(REMBG_CROPPED_IMG_DIR):
        shutil.rmtree(REMBG_CROPPED_IMG_DIR)
    os.makedirs(REMBG_CROPPED_IMG_DIR, exist_ok=True)
    
    # rembg session
    rembg_session = new_session()

    # Location of data.yaml for training
    training_abs_path = os.path.abspath(os.path.join(CUR_DIR, 'tmp_dataset', 'data.yaml'))

    # Only need to go through tier 1 and 2 if new data is uploaded
    if UPLOAD_TYPE == 'new':
        annotate(is_tier1=True, rembg_session=rembg_session, tier='tier_1')
        add_dataset()
        MODEL.train(data=training_abs_path, epochs=10, patience=1, imgsz=640)

        # Annotate tier 2
        annotate(is_tier1=False, rembg_session=rembg_session, tier='tier_2')
        add_dataset()
        MODEL.train(data=training_abs_path, epochs=10, patience=1, imgsz=640)

    # Annotate tier 3
    annotate(is_tier1=False, rembg_session=rembg_session, tier='tier_3')

    create_dataset_yaml()
    MODEL.save(os.path.join(CUR_DIR, 'tmp_general_model.pt'))

    # Cleanup
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    if os.path.exists(NEW_DATA_DIR):
        shutil.rmtree(NEW_DATA_DIR)
    if os.path.exists(REMBG_CROPPED_IMG_DIR):
        shutil.rmtree(REMBG_CROPPED_IMG_DIR)
    if os.path.exists('runs'):
        shutil.rmtree('runs')

    send_annotation_status('DONE')

def annotate(is_tier1, rembg_session, tier):
    """Annotate images in tier"""
    tier_path = os.path.join(NEW_DATA_DIR, tier)
    for img_file in os.listdir(tier_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):            
            # Load image
            img_path = os.path.join(tier_path, img_file)

            with Image.open(img_path) as image:
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # If not tier 1, use model to get annotations for each object in image
                if not is_tier1:
                    annotations = annotate_model(MODEL, image)
                # Otherwise, use whole image as annotation
                else:
                    rembg_img = remove(cropped_image, session=rembg_session, alpha_matting=True, alpha_matting_background_threshold=50)
                    bbox = rembg_img.getbbox()
                    annotations = [(NEW_CLASS_ID, bbox)]

                    cropped_image = rembg_img.crop(bbox)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    cropped_image.save(os.path.join(REMBG_CROPPED_IMG_DIR, f'{timestamp}.png'))

                if len(annotations) > 0:
                    # Save to dataset name
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    image_name = f'{base_name}.{timestamp}'
                    
                    # Save to new dataset
                    image.save(os.path.join(ANNOTATIONS_DIR, 'images', f'{image_name}.png'))

                    label_path = os.path.join(ANNOTATIONS_DIR, 'labels', f'{image_name}.txt')
                    with open(label_path, 'w') as f:
                        for class_id, bbox in annotations:
                            class_id = int(class_id)
                            x_center, y_center, bbox_width, bbox_height = bbox_to_yolo(bbox, image.width, image.height)
                            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def bbox_to_yolo(bbox, width, height):
    """Helper function to convert bbox to YOLO format"""
    x_center = (bbox[0] + bbox[2]) / (2 * width)
    y_center = (bbox[1] + bbox[3]) / (2 * height)
    bbox_width = (bbox[2] - bbox[0]) / width
    bbox_height = (bbox[3] - bbox[1]) / height

    return x_center, y_center, bbox_width, bbox_height

def annotate_model(model, image):
    """Annotate image using model"""
    items = []
    results = model(image)
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        items.append((class_id, [x1, y1, x2, y2]))

    return items

def create_dataset_yaml():
    # Create data.yaml
    global YAML_CONTENT, NEW_CLASS_ID, CLASS_NAME, MODEL, NEW_CLASS
    if YAML_CONTENT is None:
        if NEW_CLASS:
            yaml_content = f"""train: ./train/images
val: ./valid/images
test: ./test/images

nc: {NC}
names: {list(MODEL.names.values()) + [CLASS_NAME]}
"""
        else:
            yaml_content = f"""train: ./train/images
val: ./valid/images
test: ./test/images

nc: {NC}
names: {list(MODEL.names.values())}"""

    if os.path.exists(DATASET_DIR):
        with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
    if os.path.exists(ANNOTATIONS_DIR):
        with open(os.path.join(ANNOTATIONS_DIR, 'data.yaml'), 'w') as f:
            f.write(yaml_content)

def add_dataset():
    """Adds random existing images to training set to avoid overfittin on new data"""
    # Copy dataset to tmp dataset
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    shutil.copytree(EXISTING_DATA_DIR, DATASET_DIR)

    # Create data.yaml
    if os.path.exists(os.path.join(DATASET_DIR, 'data.yaml')):
        os.remove(os.path.join(DATASET_DIR, 'data.yaml'))
    create_dataset_yaml()

    # Get existing images
    existing_images = glob.glob(os.path.join(DATASET_DIR, 'train', 'images', '*'))

    # Get cropped images
    cropped_images = glob.glob(os.path.join(REMBG_CROPPED_IMG_DIR, '*'))

    def calculate_overlap(rect1, rect2):
        # Calculate intersection
        x_left = max(rect1[0], rect2[0])
        y_top = max(rect1[1], rect2[1])
        x_right = min(rect1[2], rect2[2])
        y_bottom = min(rect1[3], rect2[3])
        
        # Check if intersection exists
        if x_right > x_left and y_bottom > y_top:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate areas of both rectangles
            area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
            area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
            
            # Calculate overlap percentage
            overlap_percentage = intersection_area / min(area1, area2) * 100
            return overlap_percentage
        
        return 0

    def resize_image(cropped_img, base_img, min_ratio=0.2, max_ratio=0.4):
        scale_factor = random.uniform(min_ratio, max_ratio)

        max_width = int(base_img.width * scale_factor)
        max_height = int(base_img.height * scale_factor)

        width_ratio = max_width / cropped_img.width
        height_ratio = max_height / cropped_img.height

        scale_factor = min(width_ratio, height_ratio)

        new_width = int(cropped_img.width * scale_factor)
        new_height = int(cropped_img.height * scale_factor)

        return cropped_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Limit number of images to paste
    max_images = 500
    cur_images = 0
    overlap_threshold = 0
    
    # Temp existing images for finding a random one to paste onto
    available_images = existing_images.copy()
    while cur_images < max_images:
        for cropped_img_path in cropped_images:
            cropped_img_base = Image.open(cropped_img_path)

            try:
                # Try to find a suitable image to paste onto
                max_attempts = 100
                for _ in range(max_attempts):
                    # If no more available images, reset
                    if not available_images:
                        available_images = existing_images.copy()
                        
                    # Get random base image
                    base_img_path = random.choice(available_images)
                    available_images.remove(base_img_path)
                    base_img = Image.open(base_img_path)

                    cropped_img = resize_image(cropped_img_base, base_img)

                    base_label_filename = os.path.splitext(os.path.basename(base_img_path))[0] + '.txt'
                    base_label_path = os.path.join(DATASET_DIR, 'train', 'labels', base_label_filename)

                    existing_labels = []
                    with open(base_label_path, 'r') as f:
                        existing_labels = [list(map(float, line.strip().split())) for line in f.readlines()]
                    
                    # Choose a random position to paste the image to
                    max_x = base_img.width - cropped_img.width
                    max_y = base_img.height - cropped_img.height
                    
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)

                    # Check if the cropped image overlaps with any existing labels
                    acceptable_overlap = False
                    cropped_rect = (x, y, x + cropped_img.width, y + cropped_img.height)

                    for label in existing_labels:
                        class_id, x_center, y_center, bbox_width, bbox_height = label
                        label_x1 = int((x_center - bbox_width/2) * base_img.width)
                        label_y1 = int((y_center - bbox_height/2) * base_img.height)
                        label_x2 = int((x_center + bbox_width/2) * base_img.width)
                        label_y2 = int((y_center + bbox_height/2) * base_img.height)

                        # If overlap is less than threshold, accept
                        if calculate_overlap(cropped_rect, (label_x1, label_y1, label_x2, label_y2)) <= overlap_threshold:
                            acceptable_overlap = True
                        else:
                            acceptable_overlap = False
                            break
            except Exception as e:
                print(e)
                print(base_img_path)
            
            # If a suitable image was found, paste the cropped image onto it and update labels
            if acceptable_overlap:
                # Create a copy of the base image to modify
                base_img_copy = base_img.copy()
                base_img_copy.paste(cropped_img, (x, y), cropped_img)
                
                # Save modified image
                base_img_copy.save(base_img_path)
                
                # Update label file
                with open(base_label_path, 'a+') as f:
                    # Convert bbox to YOLO format
                    x_center = (x + cropped_img.width/2) / base_img.width
                    y_center = (y + cropped_img.height/2) / base_img.height
                    bbox_width = cropped_img.width / base_img.width
                    bbox_height = cropped_img.height / base_img.height

                    # Ensure writing is on a new line
                    f.seek(0, 2)
                    f.seek(f.tell() - 1)
                    last_char = f.read(1)
                    if last_char != '\n':
                        f.write('\n')
                    
                    f.write(f"{NEW_CLASS_ID} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                    cur_images += 1
        
        overlap_threshold += 2

        if overlap_threshold > 50:
            print('Could not find enough suitable images to paste onto, pasted: ', cur_images)
            break
        if cur_images >= max_images:
            print('Pasted onto enough images')
            break
    
    dataset_allotter = DatasetAllocator()
    annotated = glob.glob(os.path.join(ANNOTATIONS_DIR, 'images', '*'))
    for img_path in annotated:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        image_path, label_path = dataset_allotter.get_paths(img_name)
        shutil.copy(img_path, image_path)
        shutil.copy(os.path.join(ANNOTATIONS_DIR, 'labels', img_name + '.txt'), label_path)

class DatasetAllocator:
    """Helper class to allocate images to training, validation, and test sets"""
    def __init__(self):
        self.dataset_path = DATASET_DIR
        self.count = 0
    
    def get_paths(self, filename):
        self.count += 1
        if self.count % 9 == 0 or self.count % 10 == 0:
            image_path = os.path.join(self.dataset_path, 'valid', 'images', filename + '.png')
            label_path = os.path.join(self.dataset_path, 'valid', 'labels', filename + '.txt')
        else:
            image_path = os.path.join(self.dataset_path, 'train', 'images', filename + '.png')
            label_path = os.path.join(self.dataset_path, 'train', 'labels', filename + '.txt')
        
        return image_path, label_path

if __name__ == '__main__':
    def send_annotation_status(status):
        print(status)

    annotate_images(upload_type='new', class_name='test', send_annotation_status=send_annotation_status)