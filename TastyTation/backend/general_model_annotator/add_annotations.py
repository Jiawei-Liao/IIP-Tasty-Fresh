import os
import shutil
import glob

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = './dataset'
ANNOTATIONS_DIR = os.path.join(CUR_DIR, 'annotations')

def add_annotations(send_annotation_status):
    # Replace data.yaml
    source_yaml_path = os.path.join(ANNOTATIONS_DIR, 'data.yaml')
    target_yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    shutil.copy(source_yaml_path, target_yaml_path)

    # Allocate data to dataset
    dataset_allocator = DatasetAllocator()
    for image_file in glob.glob(os.path.join(ANNOTATIONS_DIR, 'images', '*')):
        filename = os.path.splitext(os.path.basename(image_file))[0]
        image_path, label_path = dataset_allocator.get_paths(filename)

        # Copy image and label
        shutil.move(image_file, image_path)
        shutil.move(os.path.join(ANNOTATIONS_DIR, 'labels', filename + '.txt'), label_path)

    # Cleanup
    shutil.rmtree(ANNOTATIONS_DIR)

    # Send annotation status
    send_annotation_status('NOT STARTED')

class DatasetAllocator:
    '''Helper class to allocate images to training, validation, and test sets'''
    def __init__(self):
        self.dataset_path = DATASET_DIR
        self.count = 0
    
    def get_paths(self, filename):
        self.count += 1
        if self.count % 8 == 0 or self.count % 9 == 0:
            image_path = os.path.join(self.dataset_path, 'valid', 'images', filename + '.png')
            label_path = os.path.join(self.dataset_path, 'valid', 'labels', filename + '.txt')
        elif self.count % 10 == 0:
            image_path = os.path.join(self.dataset_path, 'test', 'images', filename + '.png')
            label_path = os.path.join(self.dataset_path, 'test', 'labels', filename + '.txt')
        else:
            image_path = os.path.join(self.dataset_path, 'train', 'images', filename + '.png')
            label_path = os.path.join(self.dataset_path, 'train', 'labels', filename + '.txt')
        
        return image_path, label_path