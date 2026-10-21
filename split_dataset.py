import tensorflow as tf

# Dataset path
dataset_path = "Freshness Fish Dataset"

# Image settings
image_size = (224, 224)
batch_size = 32
seed = 42

print("Loading training dataset...")

# Load 80% of the images for training
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

print("\nLoading validation dataset...")

# Load the remaining 20% for validation
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

print("\nDataset splitting completed!")

print("Classes:", train_dataset.class_names)
print("Training batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))

print("\nDataset is ready for model training!")