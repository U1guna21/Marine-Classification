from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
model = load_model("inceptionV3.h5", compile=False)
IMG_SIZE = 75
fish_classes = ['Striped Red Mullet', 'Red Sea Bream', 'Black Sea Sprat', 'Trout', 'Gilt-Head Bream', 'Shrimp', 'Red Mullet', 'Hourse Mackerel', 'Sea Bass']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the user
    file = request.files['image']
    image = file.read()

    # Convert the image to an array and resize it
    img_array = np.fromstring(image, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = np.array(img) / 255.0

    # Make a prediction with the model
    prediction = model.predict(np.array([img]))
    predicted_class = fish_classes[np.argmax(prediction)]

    # Convert the image to base64 for display in the result page
    image_b64 = base64.b64encode(image).decode('utf-8')

    # Render the result page with the prediction and image
    return render_template('result.html', prediction=predicted_class, image=image_b64)

if __name__ == '__main__':
    app.run(debug=True)
