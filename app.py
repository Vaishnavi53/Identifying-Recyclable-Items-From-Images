
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pyttsx3

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'resnet.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    # Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    
    if preds == 0:
        preds = "Put it in the organic bin. Organic waste includes food scraps, yard waste, and other biodegradable materials."
    else:
        preds = "Put it in the recyclable bin. Recyclable waste includes plastics, paper, glass, and metals."
    
    return preds

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    # About page
    return render_template('about.html')

@app.route('/a', methods=['GET'])
def index1():
    # Another index page
    return render_template('index1.html')

@app.route('/blog', methods=['GET'])
def blog():
    # Blog page
    return render_template('blog.html')

@app.route('/gallery', methods=['GET'])
def gallery():
    # Gallery page
    return render_template('gallery.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds

        # Speak the result
        speak_text(result)

        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
