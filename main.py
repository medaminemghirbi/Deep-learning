from flask import Flask, request, render_template, redirect
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained model
model_path = 'skin_disease_model.keras'  # Path to your saved model
model = tf.keras.models.load_model(model_path)

# Define the image size and class names
img_size = (450, 450)
class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis',
               'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma',
               'Tinea Ringworm Candidiasis', 'Vascular lesion']  # Replace with your actual class names


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_image(image_file):
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_disease(model, img_array):
    # Predict on the image array
    preds = model.predict(img_array)
    # Format predictions
    formatted_predictions = [f'{value:.2f}' for value in preds[0]]
    top_prob_index = np.argmax(preds[0])
    top_prob = round(preds[0][top_prob_index] * 100, 2)
    # Prepare the list of probabilities with class names
    probabilities = sorted(zip(class_names, formatted_predictions), key=lambda x: x[1], reverse=True)

    return class_names[top_prob_index], top_prob, probabilities


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        if file:
            img_array = process_image(file)
            predicted_class, top_prob, probabilities = predict_disease(model, img_array)

            # Convert image to base64 for displaying in HTML
            buffered = BytesIO()
            img_pil = Image.fromarray((img_array[0] * 255).astype(np.uint8))
            img_pil.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return render_template('result.html',
                                   img_base64=img_base64,
                                   predicted_class=predicted_class,
                                   top_prob=top_prob,
                                   probabilities=probabilities)
    return render_template('index.html')


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
