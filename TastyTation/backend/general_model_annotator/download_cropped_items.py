import io
import zipfile
import os
from PIL import Image
from datetime import datetime

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

LABELS_DIR = os.path.join(CUR_DIR, 'annotations', 'labels')
IMAGES_DIR = os.path.join(CUR_DIR, 'annotations', 'images')

def download_cropped_items():
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for label_file in os.listdir(LABELS_DIR):
            if label_file.endswith('.txt'):
                image_file = label_file[:-4] + '.png'
                image_path = os.path.join(IMAGES_DIR, image_file)
                label_path = os.path.join(LABELS_DIR, label_file)

                if not os.path.exists(image_path):
                    print(f'Image {image_path} not found, skipping...')
                    continue

                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            class_id, x_center, y_center, width, height = map(float, parts)

                            # Convert YOLO format to pixel coordinates
                            x_center = int(x_center * img_width)
                            y_center = int(y_center * img_height)
                            width = int(width * img_width)
                            height = int(height * img_height)
                            
                            x_min = max(x_center - width // 2, 0)
                            y_min = max(y_center - height // 2, 0)
                            x_max = min(x_center + width // 2, img_width)
                            y_max = min(y_center + height // 2, img_height)

                            cropped_img = img.crop((x_min, y_min, x_max, y_max))

                            base_name = label_file.split('.')[0]
                            now = datetime.now()
                            formatted_datetime = now.strftime('%Y%m%d%H%M%S') + f'{now.microsecond // 1000:03d}'

                            cropped_filename = f'data/{base_name}.{formatted_datetime}.png'

                            img_buffer = io.BytesIO()
                            cropped_img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)

                            zip_file.writestr(cropped_filename, img_buffer.getvalue())
        
    zip_buffer.seek(0)
    return zip_buffer