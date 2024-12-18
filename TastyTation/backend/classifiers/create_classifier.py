import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def create_classifier(classifier_name):
    classifier_dir = os.path.join(CUR_DIR, classifier_name)

    # Check if classifier exists
    if os.path.exists(classifier_dir):
        raise Exception(f'Classifier {classifier_name} already exists!')
    
    # Create classifier directory
    os.makedirs(classifier_dir)
    os.makedirs(os.path.join(classifier_dir, 'dataset'))
    os.makedirs(os.path.join(classifier_dir, 'models'))