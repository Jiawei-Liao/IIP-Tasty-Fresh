import json
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CUR_DIR, 'inconsistent_annotations.json')

def resolve_inconsistency(image_path):
    # Load inconsistent annotations
    with open(JSON_PATH, 'r') as f:
        inconsistent_annotations = json.load(f)
    
    # Remove image from inconsistent annotations
    inconsistent_annotations = [annotation for annotation in inconsistent_annotations if annotation['image_path'] != image_path]
    
    # Save inconsistent annotations
    with open(JSON_PATH, 'w') as f:
        json.dump(inconsistent_annotations, f, indent=4)

def update_inconsistent_label(image_path, label_index):
    # Load inconsistent annotations
    with open(JSON_PATH, 'r') as f:
        inconsistent_annotations = json.load(f)
    
    # Update label index
    for annotation in inconsistent_annotations:
        if annotation['image_path'] == image_path:
            # Remove index value
            annotation['dataset_inconsistency_index'].remove(label_index)

            # Decrement subsequent indexes
            annotation['dataset_inconsistency_index'] = [
                idx - 1 if idx > label_index else idx
                for idx in annotation['dataset_inconsistency_index']
            ]
    
    # Save inconsistent annotations
    with open(JSON_PATH, 'w') as f:
        json.dump(inconsistent_annotations, f, indent=4)