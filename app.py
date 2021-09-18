import base64
import numpy as np
import io
from PIL import Image
from flask import Flask,jsonify, render_template, request
from flask_cors import CORS
import tensorflow as tf
import os

model = tf.keras.models.load_model('dogVcat_model.h5')

app = Flask(__name__)
CORS(app)

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    print(image.shape)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def mainpage():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(100, 100))
    prediction = model.predict(processed_image).tolist()
    print(prediction[0][0])
    response = {
        'prediction': {
            'result': 'Cat' if prediction[0][0] == 1 else 'Dog',
        }
    }
    return jsonify(response)