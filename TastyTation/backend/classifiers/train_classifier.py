import threading
import os
from ultralytics import YOLO
from datetime import datetime

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def train_classifier(classifier_name, send_classifier_training_status_route):
    threading.Thread(target=train_classifier_thread, args=(classifier_name, send_classifier_training_status_route)).start()

def train_classifier_thread(classifier_name, send_classifier_training_status_route):
    def status_callback(trainer):
        epoch = trainer.epoch + 1
        epochs = trainer.epochs
        metrics = trainer.metrics

        data = {
            'Epoch': epoch,
            'Total Epochs': epochs,
            'metrics': {
                'Accuracy': f'{metrics['metrics/accuracy_top1'] * 100:.2f}%',
                'Train Loss': f'{trainer.loss.item():.4f}',
                'Validation Loss': f'{metrics['val/loss']:.4f}',
            }
        }

        send_classifier_training_status_route(classifier_name, data)
    
    model = YOLO('yolo11s-cls.pt')
    model.add_callback('on_train_epoch_end', status_callback)
    model.train(data=os.path.join(CUR_DIR, classifier_name, 'dataset'), epochs=30, imgsz=640, project=os.path.join(CUR_DIR, classifier_name, 'models'), name=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model.save(os.path.join(CUR_DIR, classifier_name, 'models', f'{classifier_name}.pt'))