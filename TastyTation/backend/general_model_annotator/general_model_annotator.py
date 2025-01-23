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
    '''Save uploaded images to disk'''
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
MODEL = None
NEW_CLASS_ID = None

def annotate_images(upload_type, class_name, send_annotation_status):
    send_annotation_status('STARTED')

    # Setup model and variables
    global MODEL, NEW_CLASS_ID
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(CUR_DIR, '..', 'detection_model', 'models', 'general_model.pt') if os.path.join(CUR_DIR, '..', 'detection_model', 'models', 'general_model.pt') else 'yolo11m.pt'
    MODEL = YOLO(model_path).to(device)

    # If class name already exists, use that existing ID, otherwise, it's the next unused ID
    if class_name in MODEL.names.values():
        NEW_CLASS_ID = list(MODEL.names.values()).index(class_name)
    else:
        NEW_CLASS_ID = len(MODEL.names)
    
    # Reset annotations directory
    annotation_path = os.path.join(CUR_DIR, 'annotations')
    if os.path.exists(annotation_path):
        shutil.rmtree(annotation_path)
    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(annotation_path, subdir), exist_ok=True)
        
    # Create data.yaml in annotations
    # If uploaded images are of type existing or the class name already exists, create data.yaml without new class name
    if upload_type == 'existing' or class_name in MODEL.names.values():
        yaml_content = f'''train: ./train/images
val: ./valid/images
test: ./test/images

nc: {len(MODEL.names)}
names: {list(MODEL.names.values())}
'''
    # Otherwise, nc needs to increase by 1 and class name needs to be appended to names
    else:
        yaml_content = f'''train: ./train/images
val: ./valid/images
test: ./test/images

nc: {len(MODEL.names) + 1}
names: {list(MODEL.names.values()) + [class_name]}'''

    with open(os.path.join(ANNOTATIONS_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    # Reset cropped images directory
    if os.path.exists(REMBG_CROPPED_IMG_DIR):
        shutil.rmtree(REMBG_CROPPED_IMG_DIR)
    os.makedirs(REMBG_CROPPED_IMG_DIR, exist_ok=True)
    
    # rembg session
    rembg_session = new_session()

    # Location of data.yaml for training
    training_abs_path = os.path.abspath(os.path.join(CUR_DIR, 'tmp_dataset', 'data.yaml'))

    # Only need to go through tier 1 and 2 if new data is uploaded
    if upload_type == 'new':
        annotate(is_tier1=True, rembg_session=rembg_session, tier='tier_1')
        add_dataset()
        MODEL.train(data=training_abs_path, epochs=5, imgsz=640)

        # Annotate tier 2
        annotate(is_tier1=False, rembg_session=rembg_session, tier='tier_2')
        add_dataset()
        MODEL.train(data=training_abs_path, epochs=5, imgsz=640)

    # Annotate tier 3
    annotate(is_tier1=False, rembg_session=rembg_session, tier='tier_3')

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
    '''Annotate images in tier'''
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
                    rembg_img = remove(image, session=rembg_session, alpha_matting=True, alpha_matting_background_threshold=50)
                    bbox = rembg_img.getbbox()
                    annotations = [(NEW_CLASS_ID, bbox)]

                    cropped_image = rembg_img.crop(bbox)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    cropped_image.save(os.path.join(REMBG_CROPPED_IMG_DIR, f'{timestamp}.png'))

                if len(annotations) > 0 or True:
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
                            f.write(f'{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n')

def bbox_to_yolo(bbox, width, height):
    '''Helper function to convert bbox to YOLO format'''
    x_center = (bbox[0] + bbox[2]) / (2 * width)
    y_center = (bbox[1] + bbox[3]) / (2 * height)
    bbox_width = (bbox[2] - bbox[0]) / width
    bbox_height = (bbox[3] - bbox[1]) / height

    return x_center, y_center, bbox_width, bbox_height

def annotate_model(model, image):
    '''Annotate image using model'''
    items = []
    results = model(image)
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        items.append((class_id, [x1, y1, x2, y2]))

    return items

def add_dataset():
    '''Adds random existing images to training set to avoid overfitting on new data'''
    # Copy dataset to tmp dataset
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    shutil.copytree(EXISTING_DATA_DIR, DATASET_DIR)

    # Create data.yaml
    if os.path.exists(os.path.join(DATASET_DIR, 'data.yaml')):
        os.remove(os.path.join(DATASET_DIR, 'data.yaml'))
    shutil.copy(os.path.join(ANNOTATIONS_DIR, 'data.yaml'), os.path.join(DATASET_DIR, 'data.yaml'))
    
    # Add new annotation data to train
    dataset_allotter = DatasetAllocator()
    annotated = glob.glob(os.path.join(ANNOTATIONS_DIR, 'images', '*'))
    for img_path in annotated:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        image_path, label_path = dataset_allotter.get_paths(img_name)
        shutil.copy(img_path, image_path)
        shutil.copy(os.path.join(ANNOTATIONS_DIR, 'labels', img_name + '.txt'), label_path)
    
    def calculate_overlap(rect1, rect2):
        # Calculate intersection
        x_left = max(rect1[0], rect2[0])
        y_top = max(rect1[1], rect2[1])
        x_right = min(rect1[2], rect2[2])
        y_bottom = min(rect1[3], rect2[3])
        
        if x_right > x_left and y_bottom > y_top:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
            area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
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
    
    # Get existing images
    existing_images = glob.glob(os.path.join(DATASET_DIR, 'train', 'images', '*'))

    # Get cropped images
    cropped_images = glob.glob(os.path.join(REMBG_CROPPED_IMG_DIR, '*'))
    
    max_images = 500
    cur_images = 0
    overlap_threshold = 0
    max_overlap_threshold = 20
    
    while cur_images < max_images and overlap_threshold <= max_overlap_threshold:
        # Get unused cropped images
        available_cropped_images = cropped_images[:]
        
        if not available_cropped_images:
            available_cropped_images = cropped_images[:]
            
        for cropped_img_path in available_cropped_images:
            if cur_images >= max_images:
                break
                
            cropped_img_base = Image.open(cropped_img_path)
            available_images = existing_images.copy()
            success = False
            
            try:
                # Try to find a suitable image to paste onto
                max_attempts = 100
                for attempt in range(max_attempts):
                    if not available_images:
                        break
                        
                    base_img_path = random.choice(available_images)
                    available_images.remove(base_img_path)
                    base_img = Image.open(base_img_path)
                    cropped_img = resize_image(cropped_img_base, base_img)

                    base_label_filename = os.path.splitext(os.path.basename(base_img_path))[0] + '.txt'
                    base_label_path = os.path.join(DATASET_DIR, 'train', 'labels', base_label_filename)

                    with open(base_label_path, 'r') as f:
                        existing_labels = [list(map(float, line.strip().split())) for line in f.readlines()]
                    
                    max_x = base_img.width - cropped_img.width
                    max_y = base_img.height - cropped_img.height
                    
                    if max_x <= 0 or max_y <= 0:
                        continue
                        
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)

                    cropped_rect = (x, y, x + cropped_img.width, y + cropped_img.height)
                    acceptable_overlap = True

                    for label in existing_labels:
                        class_id, x_center, y_center, bbox_width, bbox_height = label
                        label_x1 = int((x_center - bbox_width/2) * base_img.width)
                        label_y1 = int((y_center - bbox_height/2) * base_img.height)
                        label_x2 = int((x_center + bbox_width/2) * base_img.width)
                        label_y2 = int((y_center + bbox_height/2) * base_img.height)

                        if calculate_overlap(cropped_rect, (label_x1, label_y1, label_x2, label_y2)) > overlap_threshold:
                            acceptable_overlap = False
                            break

                    if acceptable_overlap:
                        base_img_copy = base_img.copy()
                        base_img_copy.paste(cropped_img, (x, y), cropped_img)
                        base_img_copy.save(base_img_path)
                        
                        with open(base_label_path, 'a+') as f:
                            x_center = (x + cropped_img.width/2) / base_img.width
                            y_center = (y + cropped_img.height/2) / base_img.height
                            bbox_width = cropped_img.width / base_img.width
                            bbox_height = cropped_img.height / base_img.height

                            f.seek(0, 2)
                            f.seek(f.tell() - 1)
                            last_char = f.read(1)
                            if last_char != '\n':
                                f.write('\n')
                            
                            f.write(f'{NEW_CLASS_ID} {x_center} {y_center} {bbox_width} {bbox_height}\n')

                        cur_images += 1
                        available_cropped_images.remove(cropped_img_path)
                        success = True
                        print(f'Pasted {cropped_img_path} to {base_img_path} (Attempt {attempt + 1})')
                        break
                        
            except Exception as e:
                print(f'Error processing {base_img_path}: {e}')
                continue
                
            if not success:
                print(f'Failed to place {cropped_img_path} after {max_attempts} attempts')
                
        if cur_images < max_images:
            overlap_threshold += 2
            print(f'Increasing overlap threshold to {overlap_threshold}')
    
    print(f'Final results: Pasted {cur_images} images with final overlap threshold {overlap_threshold}')

class DatasetAllocator:
    '''Helper class to allocate images to training, validation, and test sets'''
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