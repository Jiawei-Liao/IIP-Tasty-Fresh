from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS

import os

from general_model_annotator.general_model_annotator import save_uploaded_images
from general_model_annotator.get_annotations import get_general_annotations

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

@app.route('/images/<path:full_path>')
def get_image(full_path):
    path, filename = os.path.split(full_path)
    return send_from_directory(path, filename)

@app.route('/api/edit-labels/<path:full_path>/labels', methods=['POST'])
def edit_labels(full_path):
    path, filename = os.path.split(full_path)

"""Endpoint for getting uploaded images for general model and annotates them"""
@app.route('/api/annotate', methods=['POST'])
def annotate():
    try:
        upload_type = request.form.get('uploadType')
        class_name = request.form.get('className', '')
        images = request.files.getlist('images')

        save_uploaded_images(images, upload_type, class_name, send_annotation_status)

        return jsonify({'message': 'Images uploaded successfully!'}), 200
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Annotation status flag
ANNOTATION_STATUS = 'DONE'
"""Endpoint for getting annotation status"""
def send_annotation_status(status):
    # Update global annotation status
    global ANNOTATION_STATUS
    ANNOTATION_STATUS = status

    # Get general annotations
    annotations = get_general_annotations()

    # Send annotation status
    socketio.emit('annotation_status', {'status': status, 'annotations': annotations})

"""Endpoint for getting annotations created by the general model"""
@app.route('/api/get-annotations', methods=['GET'])
def get_annotations():
    annotations, new_annotation_classes = get_general_annotations()
    return jsonify({'status': ANNOTATION_STATUS, 'annotations': annotations, 'new_annotation_classes': new_annotation_classes}), 200

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
