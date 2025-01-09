import os
import threading
from ultralytics import YOLO
from datetime import datetime

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def train_detection_model(send_training_status_route):
    threading.Thread(target=train_detection_model_thread, args=(send_training_status_route,)).start()

def train_detection_model_thread(send_training_status_route):
    def status_callback(trainer):
        epoch = trainer.epoch + 1
        epochs = trainer.epochs
        metrics = trainer.metrics

        data = {
            'Epoch': epoch,
            'Total Epochs': epochs,
            'metrics': {
                'Precision': f'{metrics['metrics/precision(B)'] * 100:.2f}%',
                'Recall': f'{metrics['metrics/recall(B)'] * 100:.2f}%',
                'mAP50': f'{metrics['metrics/mAP50(B)'] * 100:.2f}%',
                'mAP50-95': f'{metrics['metrics/mAP50-95(B)'] * 100:.2f}%',
                'Box Loss': f'{metrics['val/box_loss']:.4f}',
                'CLS Loss': f'{metrics['val/cls_loss']:.4f}',
                'DFL Loss': f'{metrics['val/dfl_loss']:.4f}'
            }
        }

        send_training_status_route('detection', data)

    model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    model = YOLO('yolo11m.pt')
    model.add_callback('on_train_epoch_end', status_callback)
    model.train(data=os.path.join(CUR_DIR, '..', 'dataset_', 'data.yaml'), epochs=30, imgsz=640, project=os.path.join(CUR_DIR, 'training'), name=model_name)
    model.save(os.path.join(CUR_DIR, 'models', f'{model_name}.pt'))
