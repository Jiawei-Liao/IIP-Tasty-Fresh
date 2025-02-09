import os
import io
import zipfile
from PIL import Image
from ultralytics import YOLO

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CUR_DIR, '..', 'detection_model', 'models', 'general_model.pt') if os.path.join(CUR_DIR, '..', 'detection_model', 'models', 'general_model.pt') else 'yolo11m.pt'
CONFIDENCE_THRESHOLD = 0.8

def segment_images(images):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        model = YOLO(MODEL_PATH)
        class_names = model.names

        for image in images:
            img = Image.open(image)
            img_name = image.filename

            results = model(img, verbose=False)

            for idx, detection in enumerate(results[0].boxes.data):
                x1, y1, x2, y2, confidence, class_id = detection.tolist()
                class_name = class_names[int(class_id)]

                target_dir = 'unsure' if confidence < CONFIDENCE_THRESHOLD else class_name

                cropped_img = img.crop((x1, y1, x2, y2))

                cropped_filename = f'data/{target_dir}/{os.path.splitext(os.path.basename(img_name))[0]}.{idx}.png'

                img_buffer = io.BytesIO()
                cropped_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                zip_file.writestr(cropped_filename, img_buffer.getvalue())
        
    zip_buffer.seek(0)
    return zip_buffer