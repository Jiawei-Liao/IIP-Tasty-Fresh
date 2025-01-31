import os
import shutil
from pathlib import Path

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def rename_class(classifier_name, old_class_name, new_class_name):
    splits = ['train', 'val', 'test']

    for split in splits:
        old_path = os.path.join(CUR_DIR, classifier_name, 'dataset', split, old_class_name)
        new_path = os.path.join(CUR_DIR, classifier_name, 'dataset', split, new_class_name)
        if os.path.exists(new_path):
            for item in os.listdir(old_path):
                old_item_path = os.path.join(old_path, item)
                new_item_path = os.path.join(new_path, item)

                if os.path.exists(new_item_path):
                    base, ext = os.path.splitext(new_item_path)
                    counter = 1
                    while os.path.exists(new_item_path):
                        new_name = f'{base}_{counter}{ext}'
                        new_item_path = os.path.join(new_path, new_name)
                        counter += 1
                
                shutil.move(old_item_path, new_item_path)
            
            shutil.rmtree(old_path)
        else:
            os.rename(old_path, new_path)