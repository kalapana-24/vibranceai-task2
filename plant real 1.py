# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:01:21 2024

@author: shiva
"""

# app.py

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('plant leaf disease')

# Dictionary to label all diseases
class_labels = {
    0: 'Disease 1',  # Replace with actual class labels
    1: 'Disease 2',
    2: 'Disease 3',
    # Add all your class labels
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions, axis=1)[0]
            result = class_labels[class_idx]
            
            return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)