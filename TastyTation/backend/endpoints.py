from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS

import os

from general_model_annotator.general_model_annotator import save_uploaded_images
from general_model_annotator.get_annotations import get_general_annotations
from general_model_annotator.add_annotations import add_annotations

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

@app.route('/images/<path:full_path>')
def get_image_route(full_path):
    path, filename = os.path.split(full_path)
    return send_from_directory(path, filename)

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

"""Endpoint for getting uploaded images for general model and annotates them"""
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

# Annotation status flag
ANNOTATION_STATUS = 'DONE' if os.path.exists('./general_model_annotator/annotations') else 'NOT STARTED'
"""Endpoint for getting annotation status"""
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
        print(f"Error sending annotation status: {e}")
        socketio.emit('annotation_status', {'status': 'ERROR', 'annotations': [], 'new_annotation_classes': []})

"""Endpoint for getting annotations created by the general model"""
@app.route('/api/get-annotations', methods=['GET'])
def get_annotations_route():
    if ANNOTATION_STATUS != 'DONE':
        return jsonify({'status': ANNOTATION_STATUS, 'annotations': [], 'new_annotation_classes': []}), 200
    
    annotations, new_annotation_classes = get_general_annotations()
    return jsonify({'status': ANNOTATION_STATUS, 'annotations': annotations, 'new_annotation_classes': new_annotation_classes}), 200

@app.route('/api/add-annotations', methods=['POST'])
def add_annotations_route():
    add_annotations(send_annotation_status_route)
    return jsonify({'message': 'Annotations added successfully!'}), 200
    
if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
