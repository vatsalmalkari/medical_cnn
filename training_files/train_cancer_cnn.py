import tensorflow as tf
from keras import layers, models, applications, optimizers, losses, callbacks
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. Configuration and Data Loading ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_data(path, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    ).map(lambda x, y: (layers.Rescaling(1./127.5, offset=-1)(x), y))

train_ds = load_data('data/cancer_xray/training', shuffle=True)
test_ds = load_data('data/cancer_xray/testing', shuffle=False)

# --- 2. Model Architecture ---
# Base model with pre-trained weights
base_model = applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

#  full model with data augmentation and a custom head
model = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# --- 3. Two-Phase Training Function ---
def train_phase(trainable, lr, epochs, callbacks_list):
    # Set the base model's layers to be trainable or not
    base_model.trainable = trainable
    
    # If fine-tuning, freeze all but the last 20 layers
    if trainable:
        for layer in base_model.layers[:-20]:
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.CategoricalFocalCrossentropy(gamma=1),
        metrics=['accuracy']
    )
    
    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks_list
    )

# Define callbacks for both phases
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
    callbacks.ModelCheckpoint('best_brain_tumor_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# Phase 1: Feature extraction 
print("--- Starting Phase 1: Feature Extraction ---")
train_phase(trainable=False, lr=1e-3, epochs=15, callbacks_list=callbacks_list)

# Phase 2: Fine-tuning
print("\n--- Starting Phase 2: Fine-Tuning ---")
train_phase(trainable=True, lr=1e-5, epochs=10, callbacks_list=callbacks_list)

# --- 5. Final Evaluation ---
# Load the best model saved by the ModelCheckpoint callback
model = tf.keras.models.load_model('best_brain_tumor_model.keras')

y_true = np.argmax(np.concatenate([y for _, y in test_ds], axis=0), axis=1)
y_pred = np.argmax(model.predict(test_ds), axis=1)

print("\n--- Final Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

# --- 6. Plot Confusion Matrix and Save Model ---
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Final Confusion Matrix for Cancer Detection')
plt.savefig('final_cancer_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()