import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_classifier(classifier_name):
    classifier_path = os.path.join(CUR_DIR, classifier_name, 'models', f'{classifier_name}.pt')
    classifier = classifier_path if os.path.exists(classifier_path) else None
    return classifier