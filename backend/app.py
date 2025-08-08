import warnings
warnings.filterwarnings("ignore", message=".*Protobuf gencode.*")

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load models
pneumonia_model = tf.keras.models.load_model('models/final_pneumonia_model.keras')
cancer_model = tf.keras.models.load_model('models/best_brain_tumor_model.keras')

# Reasoning map for user-friendly explanations
reasoning_map = {
    'PNEUMONIA': "The X-ray shows cloudy or white areas in the lungs, which can mean an infection like pneumonia.",
    'NORMAL': "The lungs look clear and normal. There are no signs of infection.",
    'glioma': "The MRI shows a fuzzy or uneven spot inside the brain that looks like a glioma tumor.",
    'meningioma': "The MRI shows a round area near the outer edge of the brain, which is common for meningioma.",
    'pituitary': "The MRI shows a spot near the center of the brain, close to the pituitary gland.",
    'notumor': "The brain scan looks normal. There are no signs of any tumor."
}

def preprocess_image(image, model_type):
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)

    if model_type == 'pneumonia':
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    else:  # cancer
        img_array = (img_array / 127.5) - 1

    return np.expand_dims(img_array, axis=0)
app = Flask(__name__, template_folder='../templates')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')
    file = request.files['file'].read()

    if model_type == 'pneumonia':
        img = preprocess_image(file, 'pneumonia')
        pred = pneumonia_model.predict(img)[0]
        classes = ['NORMAL', 'PNEUMONIA']
    else:
        img = preprocess_image(file, 'cancer')
        pred = cancer_model.predict(img)[0]
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    predicted_class = classes[np.argmax(pred)]
    reasoning_text = reasoning_map.get(predicted_class, "No explanation available.")

    return jsonify({
        'predictions': dict(zip(classes, [float(p) for p in pred])),
        'diagnosis': predicted_class,
        'reasoning': reasoning_text
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

