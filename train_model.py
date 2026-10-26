import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Dataset path
dataset_path = "Freshness Fish Dataset"

# Basic settings
image_size = (224, 224)
batch_size = 32
seed = 42
epochs = 5

print("Preparing fish freshness dataset...")

# Load training images
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

# Load validation images
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

class_names = train_dataset.class_names
number_of_classes = len(class_names)

print("\nFish quality classes:", class_names)

# Improve data loading performance
train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE
)

# Load the pretrained MobileNetV2 model
print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

# Build the classification model
inputs = layers.Input(shape=(224, 224, 3))

# Data augmentation
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomTranslation(0.1, 0.1)(x)

# Preprocess images for MobileNetV2
x = layers.Lambda(preprocess_input)(x)

# Extract visual features
x = base_model(x, training=False)

# Classification layers
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    number_of_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# Train the model
print("\nStarting model training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)
# Phase 2: fine-tune the top layers of MobileNetV2
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=8
)
# Save the trained model
model.save("fish_freshness_model_v4.keras")

print("\nModel training completed!")
print("Model saved as fish_freshness_model_v4.keras")

# Display final metrics
print(
    "Final training accuracy:",
    round(history_ft.history["accuracy"][-1] * 100, 2),
    "%"
)

print(
    "Final validation accuracy:",
    round(history_ft.history["val_accuracy"][-1] * 100, 2),
    "%"
)

print("\nFish freshness model is ready!")