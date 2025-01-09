import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_detection_models():
    models_dir = os.path.join(CUR_DIR, 'models')
    if os.path.exists(models_dir):
        models = sorted(
            [model for model in os.listdir(models_dir) if model.endswith('.pt')],
            key=lambda model: os.path.getctime(os.path.join(models_dir, model)),
            reverse=True
        )
        return [model.split('.')[0] for model in models]
    else:
        return []
