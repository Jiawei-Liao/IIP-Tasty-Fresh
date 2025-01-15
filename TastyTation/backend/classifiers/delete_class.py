import os
import shutil

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def delete_class(classifier_name, class_name):
    splits = ['train', 'val', 'test']

    for split in splits:
        path = os.path.join(CUR_DIR, classifier_name, 'dataset', split, class_name)
        if os.path.exists(path):
            shutil.rmtree(path)