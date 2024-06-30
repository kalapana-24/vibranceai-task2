# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:17:59 2024

@author: shiva
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load and compile the model
model = tf.keras.models.load_model(r"C:\Users\shiva\chinna\plant_disease_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # assuming data is sent as JSON
    input_data = np.array(data['input']).reshape(1, -1)  # adjust shape as needed
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    # Use 'threaded=True' instead of 'debug=True' to avoid using watchdog
    app.run(threaded=True)
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
