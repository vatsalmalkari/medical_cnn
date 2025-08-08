import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os

# --- Parameters ---
image_size = (128, 128)
batch_size = 32
seed = 42
test_dir = 'data/pneumonia_xray/test'
model_path = 'models/final_pneumonia_model.keras'
results_dir = 'results'

os.makedirs(results_dir, exist_ok=True)

# --- Load and preprocess test dataset ---
rescale_layer = tf.keras.layers.Rescaling(1./255)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
    seed=seed
)

class_names = test_dataset.class_names
test_dataset = test_dataset.map(lambda x, y: (rescale_layer(x), y))


# --- Load the trained model ---
model = tf.keras.models.load_model(model_path)

# --- Evaluate the model ---
loss, accuracy = model.evaluate(test_dataset)

# --- Save evaluation results ---
eval_path = os.path.join(results_dir, 'evaluation.txt')
with open(eval_path, 'w') as f:
    f.write("Model Evaluation Results\n")
    f.write("========================\n")
    f.write(f"Test Loss    : {loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")

# --- Get predictions and true labels ---
y_true, y_pred = [], []
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true, y_pred = np.array(y_true), np.array(y_pred)

# --- Confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(6,6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
cm_path = os.path.join(results_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()

# --- Classification report ---
target_names = ['Normal', 'Pneumonia']  # Adjust if needed
report_text = classification_report(y_true, y_pred, target_names=target_names)
report_path = os.path.join(results_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report\n")
    f.write("=====================\n")
    f.write(report_text)
