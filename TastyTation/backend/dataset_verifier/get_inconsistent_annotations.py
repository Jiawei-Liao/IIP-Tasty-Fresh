import os
import json
import yaml

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(CUR_DIR, 'inconsistent_annotations.json')

def get_inconsistent_annotations():
    if not os.path.exists(JSON_DIR):
        return []
    
    valid_annotations = []

    with open(JSON_DIR, 'r') as f:
        inconsistent_annotations = json.load(f)
    
    for annotation in inconsistent_annotations:
        try:
            image_annotations = []
            label_path = annotation['image_path'].removeprefix('/images/').replace('/images/', '/labels/')
            label_path = os.path.splitext(label_path)[0] + '.txt'

            if not os.path.exists(label_path):
                print(f'Label file not found: {label_path}')
                continue

            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x1, y1, x2, y2 = line.strip().split()
                    image_annotations.append({
                        'class_id': int(class_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
            annotation['annotations'] = image_annotations
            valid_annotations.append(annotation)
        
        except Exception as e:
            print(f"Error processing annotation {annotation}: {e}")
            continue
    
    annotation_classes = []
    yaml_path = os.path.join('dataset', 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            for i, class_name in enumerate(data['names']):
                annotation_classes.append({'id': i, 'name': class_name})

    return valid_annotations, annotation_classes