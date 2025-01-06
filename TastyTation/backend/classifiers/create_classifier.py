import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def create_classifier(classifier_name):
    classifier_dir = os.path.join(CUR_DIR, classifier_name)

    # Check if classifier exists
    if os.path.exists(classifier_dir):
        raise Exception(f'Classifier {classifier_name} already exists!')
    
    # Create classifier directory
    os.makedirs(classifier_dir)
    dataset_dir = os.path.join(classifier_dir, 'dataset')
    os.makedirs(dataset_dir)

    os.makedirs(os.path.join(dataset_dir, 'train'))
    os.makedirs(os.path.join(dataset_dir, 'val'))
    os.makedirs(os.path.join(dataset_dir, 'test'))

    os.makedirs(os.path.join(classifier_dir, 'models'))