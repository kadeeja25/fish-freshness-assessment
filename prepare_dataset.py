import os
import tensorflow as tf

# Dataset location
dataset_path = "Freshness Fish Dataset"

print("TensorFlow version:", tf.__version__)
print("\nLoading fish dataset...")

# Load images from the dataset folders
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="int",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)

print("\nDataset loaded successfully!")
print("Class names:", dataset.class_names)

# Check the first batch
for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

print("\nDataset preparation check completed!")
