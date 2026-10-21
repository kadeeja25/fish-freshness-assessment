import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Dataset settings
dataset_path = "Freshness Fish Dataset"
image_size = (224, 224)
batch_size = 32
seed = 42

print("Loading trained fish quality model...")

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    "fish_freshness_model.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

validation_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )
)

class_names = validation_dataset.class_names

print("\nClasses:", class_names)

class_counts = {name: 0 for name in class_names}

for images, labels in validation_dataset:
    for label in labels.numpy():
        class_counts[class_names[int(label)]] += 1

print("\nValidation images per class:")
for name, count in class_counts.items():
    print(f"{name}: {count}")
# Evaluate the model
loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\nModel Evaluation Results")
print("-" * 35)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Collect labels and predictions together
actual_labels = []
predicted_labels = []

for images,labels in validation_dataset:
    preds = model.predict(images, verbose=0)
    actual_labels.extend(labels.numpy())
    predicted_labels.extend(np.argmax(preds, axis=1))

actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)

# Display classification report
print("\nClassification Report")
print("-" * 35)

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

# Create confusion matrix
cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(values_format="d")
plt.title("Fish Quality Classification")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved successfully!")