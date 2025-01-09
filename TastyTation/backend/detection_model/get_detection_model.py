import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_detection_model(model_name):
    model_path = os.path.join(CUR_DIR, 'models', f'{model_name}.pt')

    if os.path.exists(model_path):
        model = model_path
    else:
        model = None

    return model