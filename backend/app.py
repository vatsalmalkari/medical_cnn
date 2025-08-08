import warnings
warnings.filterwarnings("ignore", message=".*Protobuf gencode.*")

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load models
pneumonia_model = tf.keras.models.load_model('backend/models/final_pneumonia_model.keras')
cancer_model = tf.keras.models.load_model('backend/models/best_brain_tumor_model.keras')

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
    img = image.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)

    if model_type == 'pneumonia':
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    else:  # cancer
        img_array = (img_array / 127.5) - 1

    return np.expand_dims(img_array, axis=0)

def predict(model_type, image):
    if image is None:
        return "No image uploaded", None, None

    if model_type == 'pneumonia':
        img = preprocess_image(image, 'pneumonia')
        pred = pneumonia_model.predict(img)[0]
        classes = ['NORMAL', 'PNEUMONIA']
    else:
        img = preprocess_image(image, 'cancer')
        pred = cancer_model.predict(img)[0]
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    predicted_class = classes[np.argmax(pred)]
    reasoning = reasoning_map.get(predicted_class, "No explanation available.")
    scores = {cls: float(score) for cls, score in zip(classes, pred)}

    return predicted_class, reasoning, scores

# Create Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(choices=["pneumonia", "cancer"], label="Select Model"),
        gr.Image(type="pil", label="Upload Medical Image")
    ],
    outputs=[
        gr.Textbox(label="Diagnosis"),
        gr.Textbox(label="Reasoning"),
        gr.Label(label="Prediction Scores")
    ],
    title="Medical Diagnosis CNN",
    description="Upload an image (Chest X-ray or Brain MRI) and select the appropriate model to get a diagnosis."
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

