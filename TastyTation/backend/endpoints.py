from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO
from flask_cors import CORS

import os

from general_model_annotator.general_model_annotator import save_uploaded_images
from general_model_annotator.get_annotations import get_general_annotations
from general_model_annotator.add_annotations import add_annotations

from dataset_verifier.dataset_verifier import verify_dataset
from dataset_verifier.get_inconsistent_annotations import get_inconsistent_annotations
from dataset_verifier.update_inconsistent_annotations import resolve_inconsistency
from dataset_verifier.update_inconsistent_annotations import update_inconsistent_label

from classifiers.segment_images import segment_images
from classifiers.create_classifier import create_classifier
from classifiers.get_classifiers import get_classifiers
from classifiers.add_images import add_images
from classifiers.get_classifier import get_classifier
from classifiers.train_classifier import train_classifier
from classifiers.view_classifier_classes import view_classifier_classes
from classifiers.delete_class import delete_class
from classifiers.rename_class import rename_class

from detection_model.get_detection_models import get_detection_models
from detection_model.get_detection_model import get_detection_model
from detection_model.train_detection_model import train_detection_model

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='http://localhost:3000')

''' Endpoints for editing annotations components '''
# Fetch image based on the path
@app.route('/images/<path:full_path>')
def get_image_route(full_path):
    path, filename = os.path.split(full_path)
    return send_from_directory(path, filename)

# Edit labels for an image
@app.route('/api/edit-labels', methods=['POST'])
def edit_labels_route():
    try:
        data = request.get_json()
        image_path = data['image']
        labels = data['annotations']

        label_path = image_path.replace('/images/', '', 1)
        label_path = label_path.replace('/images/', '/labels/', 1)
        root, ext = os.path.splitext(label_path)
        label_path = root + '.txt'

        with open(label_path, 'w') as f:
            for label in labels:
                class_id = label['class_id']
                bbox = label['bbox']
                bbox = ' '.join(map(str, bbox))
                annotation = f'{class_id} {bbox}\n'
                f.write(annotation)

        return jsonify({'message': 'Labels updated successfully!'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/delete-image', methods=['POST'])
def delete_image_route():
    try:
        image_path = request.form.get('imagePath')
        image_path = image_path.replace('/images/', '', 1)
        label_path = image_path.replace('/images/', '/labels/', 1)
        root, ext = os.path.splitext(label_path)
        label_path = root + '.txt'
        print(image_path)
        print(label_path)

        os.remove(image_path)
        os.remove(label_path)

        return jsonify({'message': 'Image deleted successfully!'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

''' Endpoints for getting upload images page '''
# Upload images, sending success and async annotate images
@app.route('/api/annotate', methods=['POST'])
def annotate_route():
    try:
        upload_type = request.form.get('uploadType')
        class_name = request.form.get('className', '')
        images = request.files.getlist('images')

        save_uploaded_images(images, upload_type, class_name, send_annotation_status_route)

        return jsonify({'message': 'Images uploaded successfully!'}), 200
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500

''' Endpoints for new annotations page '''
# Global annotation status
ANNOTATION_STATUS = 'DONE' if os.path.exists('./general_model_annotator/annotations') else 'NOT STARTED'

# Socket for sending annotation status
def send_annotation_status_route(status):
    # Update global annotation status
    global ANNOTATION_STATUS
    ANNOTATION_STATUS = status

    try:
        if ANNOTATION_STATUS != 'DONE':
            socketio.emit('annotation_status', {'status': ANNOTATION_STATUS, 'annotations': [], 'new_annotation_classes': []})

        # Get general annotations
        annotations, new_annotation_classes = get_general_annotations()

        # Send annotation status
        socketio.emit('annotation_status', {'status': ANNOTATION_STATUS, 'annotations': annotations, 'new_annotation_classes': new_annotation_classes})
    except Exception as e:
        print(f'Error sending annotation status: {e}')
        socketio.emit('annotation_status', {'status': 'ERROR', 'annotations': [], 'new_annotation_classes': []})

# Get annotation status
@app.route('/api/get-annotations', methods=['GET'])
def get_annotations_route():
    if ANNOTATION_STATUS != 'DONE':
        return jsonify({'status': ANNOTATION_STATUS, 'annotations': [], 'new_annotation_classes': []}), 200
    
    annotations, new_annotation_classes = get_general_annotations()
    return jsonify({'status': ANNOTATION_STATUS, 'annotations': annotations, 'new_annotation_classes': new_annotation_classes}), 200

# Signal to move new annotations to dataset
@app.route('/api/add-annotations', methods=['POST'])
def add_annotations_route():
    add_annotations(send_annotation_status_route)
    return jsonify({'message': 'Annotations added successfully!'}), 200

''' Endpoints for verify annotations page '''
# Global verification status
VERIFICATION_STATUS = 'DONE' if os.path.exists('./dataset_verifier/inconsistent_annotations.json') else 'NOT STARTED'

# Socket for sending verification status
def send_verification_status_route(status):
    # Update global verification status
    global VERIFICATION_STATUS
    VERIFICATION_STATUS = status

    try:
        if VERIFICATION_STATUS != 'DONE':
            socketio.emit('verification_status', {'status': VERIFICATION_STATUS, 'inconsistent_annotations': [], 'annotation_classes': []})
        
        # Get inconsistent annotations
        inconsistent_annotations, annotation_classes = get_inconsistent_annotations()

        # Send verification status
        socketio.emit('verification_status', {'status': VERIFICATION_STATUS, 'inconsistent_annotations': inconsistent_annotations, 'annotation_classes': annotation_classes})
    except Exception as e:
        print(f'Error sending verification status: {e}')
        socketio.emit('verification_status', {'status': 'ERROR', 'inconsistent_annotations': [], 'annotation_classes': []})

# Get verification status
@app.route('/api/get-verification', methods=['GET'])
def get_verification_route():
    if VERIFICATION_STATUS != 'DONE':
        return jsonify({'status': VERIFICATION_STATUS, 'inconsistent_annotations': [], 'annotation_classes': []}), 200
    
    inconsistent_annotations, annotation_classes = get_inconsistent_annotations()
    return jsonify({'status': VERIFICATION_STATUS, 'inconsistent_annotations': inconsistent_annotations, 'annotation_classes': annotation_classes}), 200

# Signal to verify dataset
@app.route('/api/verify-dataset', methods=['POST'])
def verify_dataset_route():
    verify_dataset(send_verification_status_route)
    return jsonify({'message': 'Dataset verified successfully!'}), 200

# Resolve inconsistencies in json by removing an image
@app.route('/api/resolve-inconsistency', methods=['POST'])
def resolve_inconsistency_route():
    image_path = request.form.get('image_path')
    resolve_inconsistency(image_path)
    return jsonify({'message': 'Inconsistency resolved successfully!'}), 200

# Remove or update inconsistent label index when a label is deleted
@app.route('/api/update-inconsistent-label', methods=['POST'])
def update_inconsistent_label_route():
    image_path = request.form.get('image_path')
    label_index = int(request.form.get('label_index'))
    update_inconsistent_label(image_path, label_index)
    return jsonify({'message': 'Inconsistent label updated successfully!'}), 200

''' Endpoints for classifiers page '''
# Segment objects within images
@app.route('/api/segment-images', methods=['POST'])
def segment_images_route():
    images = request.files.getlist('images')

    try:
        zip_buffer = segment_images(images)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='segmented_images.zip'
        )
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Create new classifier
@app.route('/api/create-classifier', methods=['POST'])
def create_classifier_route():
    classifier_name = request.form.get('classifier_name')

    try:
        create_classifier(classifier_name)
        return jsonify({'message': 'Classifier created successfully!'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Get list of classifiers
@app.route('/api/get-classifiers', methods=['GET'])
def get_classifiers_route():
    classifiers = get_classifiers()
    return jsonify({'classifiers': classifiers}), 200

# Add images to specified classifier of specified class
@app.route('/api/add-classifier-images', methods=['POST'])
def add_classifier_images_route():
    classifier_name = request.form.get('classifierName')
    class_name = request.form.get('className')
    images = request.files.getlist('images')
    add_images(classifier_name, class_name, images)
    return jsonify({'message': 'Images added to classifier successfully!'}), 200

# Socket for sending classifier training status
def send_training_status_route(classifier_name, status):
    try:
        socketio.emit(f'{classifier_name}_training_status', {'status': status})
    except Exception as e:
        print(f'Error sending classifier training status: {e}')

# Train classifier
@app.route('/api/train-classifier', methods=['POST'])
def train_classifier_route():
    classifier_name = request.form.get('classifierName')
    train_classifier(classifier_name, send_training_status_route)
    return jsonify({'message': 'Classifier training started!'}), 200

# Get classifier model
@app.route('/api/get-classifier-model', methods=['POST'])
def get_classifier_model_route():
    classifier_name = request.form.get('classifierName')
    classifier_model = get_classifier(classifier_name)
    if classifier_model:
        return send_file(
            classifier_model,
            as_attachment=True,
            download_name=f'{classifier_name}.pt'
        )
    else:
        return jsonify({'message': 'Classifier model not found!'}), 404

# View classifiers classes
@app.route('/api/view-classifier-classes', methods=['POST'])
def view_classifier_classes_route():
    classifier_name = request.form.get('classifierName')
    classes = view_classifier_classes(classifier_name)
    return jsonify({'classes': classes}), 200

@app.route('/api/delete-class', methods=['POST'])
def delete_class_route():
    classifier_name = request.form.get('classifierName')
    class_name = request.form.get('className')
    delete_class(classifier_name, class_name)
    classes = view_classifier_classes(classifier_name)
    return jsonify({'classes': classes}), 200

@app.route('/api/rename-class', methods=['POST'])
def rename_class_route():
    classifier_name = request.form.get('classifierName')
    old_class_name = request.form.get('oldClassName')
    new_class_name = request.form.get('newClassName')
    rename_class(classifier_name, old_class_name, new_class_name)
    classes = view_classifier_classes(classifier_name)
    return jsonify({'classes': classes}), 200

''' Endpoints for train page'''
# Get model to download based on model name
@app.route('/api/get-detection-model', methods=['POST'])
def get_detection_model_route():
    model_name = request.form.get('model')

    model = get_detection_model(model_name)
    if model:
        return send_file(
            model,
            as_attachment=True,
            download_name=f'{model_name}.pt'
        )
    else:
        return jsonify({'message': 'Model not found!'}), 404

# Get list of available models
@app.route('/api/get-detection-models', methods=['GET'])
def get_detection_models_route():
    models = get_detection_models()
    return jsonify({'models': models}), 200

# Train new detection model
@app.route('/api/train-detection-model', methods=['GET'])
def train_detection_model_route():
    train_detection_model(send_training_status_route)
    return jsonify({'message': 'Detection model training started!'}), 200

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
