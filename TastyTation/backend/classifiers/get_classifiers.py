import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_DIR = ['__pycache__']

def get_classifiers():
    classifiers = [d for d in os.listdir(CUR_DIR) if os.path.isdir(os.path.join(CUR_DIR, d)) and d not in EXCLUDE_DIR]
    return classifiers