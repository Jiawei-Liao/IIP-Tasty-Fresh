import os
import glob
import os
import yaml

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_general_annotations():
    '''Get annotations creayed by general model annotator'''
    annotations = []
    labels_dir = os.path.join(CUR_DIR, 'annotations', 'labels')
    images_dir = os.path.join(CUR_DIR, 'annotations', 'images')
    for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
        image_file = os.path.join(images_dir, os.path.basename(label_file).removesuffix('.txt') + '.png')

        if not os.path.exists(image_file):
            os.remove(label_file)
            continue

        image_annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                class_id, x1, y1, x2, y2 = line.strip().split()
                image_annotations.append({
                    'class_id': int(class_id),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

        annotations.append({
            'image_path': f'/images/general_model_annotator/annotations/images/{os.path.basename(image_file)}',
            'annotations': image_annotations
        })
    
    new_annotation_classes = []
    yaml_path = os.path.join(CUR_DIR, 'annotations', 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            for i, class_name in enumerate(data['names']):
                new_annotation_classes.append({'id': i, 'name': class_name})

    return annotations, new_annotation_classes