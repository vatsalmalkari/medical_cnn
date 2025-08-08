import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Rescaling, RandomFlip, RandomRotation, RandomZoom, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- File Paths and Parameters ---
image_size = (128, 128)  # Optimized image size for faster training
batch_size = 32
seed = 42
num_classes = 2  # Pneumonia and Normal

train_dir = 'data/pneumonia_xray/train'
test_dir = 'data/pneumonia_xray/test'

# --- 1. Load, Preprocess, and Augment the Data ---
rescale_layer = Rescaling(1./255)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    seed=seed
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    seed=seed
)

train_dataset = train_dataset.map(lambda image, label: (rescale_layer(image), label))
test_dataset = test_dataset.map(lambda image, label: (rescale_layer(image), label))

# --- 2. Build the CNN Model (Stage 1) ---
# Use MobileNetV2 for faster training
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the layers for initial training

# Data augmentation pipeline
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# Create your new model on top of the base model
model = Sequential([
    data_augmentation,
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Add a dropout layer for regularization
    Dense(num_classes, activation='softmax')
])

# --- 3. Compile and Train the Model (Stage 1: Initial Training) ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("--- Starting Stage 1: Initial Training ---")
model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset
)

# --- 4. Fine-Tuning Stage (Stage 2) ---
# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Freeze all but the last 20 layers for fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile the model with a very low learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("--- Starting Stage 2: Fine-Tuning with Early Stopping ---")
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_dataset,
    epochs=20, # Run for more epochs here with early stopping
    validation_data=test_dataset,
    callbacks=[early_stopping_callback]
)

# --- 5. Evaluate the Final Model ---
loss, accuracy = model.evaluate(test_dataset)
print(f"Final Pneumonia Test Accuracy after Fine-Tuning: {accuracy*100:.2f}%")

# --- 6. Save the Final Model ---
model.save('final_pneumonia_model.keras')
print("Model saved to final_pneumonia_model.keras")