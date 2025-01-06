import os
import json
from datetime import datetime

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def add_images(classifier_name, class_name, images):
    dataset_path = os.path.join(CUR_DIR, classifier_name, 'dataset')

    if not os.path.exists(os.path.join(dataset_path, 'train', class_name)):
        os.makedirs(os.path.join(dataset_path, 'train', class_name))
    
    if not os.path.exists(os.path.join(dataset_path, 'val', class_name)):
        os.makedirs(os.path.join(dataset_path, 'val', class_name))

    if not os.path.exists(os.path.join(dataset_path, 'test', class_name)):
        os.makedirs(os.path.join(dataset_path, 'test', class_name))

    dataset_allocator = DatasetAllocator(dataset_path, class_name)

    for image in images:
        metadata = json.loads(image.filename)
        filename = metadata['name']
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]

        save_path = dataset_allocator.get_paths(f'{filename}.{timestamp}.png')

        image.save(save_path)

class DatasetAllocator:
    '''Helper class to allocate images to training, validation, and test sets'''
    def __init__(self, dataset_path, class_name):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.count = 0
    
    def get_paths(self, filename):
        self.count += 1
        if self.count % 9 == 0 or self.count % 10 == 0:
            image_path = os.path.join(self.dataset_path, 'val', self.class_name, filename)
        else:
            image_path = os.path.join(self.dataset_path, 'train', self.class_name, filename)
        
        return image_path