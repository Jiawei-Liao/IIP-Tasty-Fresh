import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def view_classifier_classes(classifier_name):
    dataset_dir = os.path.join(CUR_DIR, classifier_name, 'dataset')
    splits = ['train', 'val', 'test']

    # Find all classes within the classifier
    classes = set()
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        for class_name in os.listdir(split_dir):
            classes.add(class_name)

    # Make sure that folders for each class exists for each split
    for class_name in classes:
        for split in splits:
            split_dir = os.path.join(dataset_dir, split)
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    # Get number of train, val, test images for each class
    class_counts = {}
    for class_name in classes:
        class_counts[class_name] = {}
        total_images = 0
        for split in splits:
            split_dir = os.path.join(dataset_dir, split)
            class_dir = os.path.join(split_dir, class_name)
            images_in_split = len(os.listdir(class_dir))
            class_counts[class_name][split] = images_in_split
            total_images += images_in_split
        class_counts[class_name]['total'] = total_images
    
    # Sort the dictionary by total number of images
    sorted_class_counts = dict(sorted(
        class_counts.items(),
        key=lambda x: x[1]['total'],
        reverse=True
    ))

    return sorted_class_counts