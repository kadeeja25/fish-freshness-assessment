import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

FOLDER =r"C:\Users\Hawwa Shadiya\Desktop\fish freshness\Freshness Fish Dataset\C3" # <-- paste the path of your C3 folder
CLASS_NAMES = ["C1", "C2", "C3"]

model = tf.keras.models.load_model(
    "fish_freshness_model.keras",
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False,
    compile=False,
)

wrong = 0
for name in sorted(os.listdir(FOLDER)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = Image.open(os.path.join(FOLDER, name)).convert("RGB").resize((224, 224))
    x = np.array(img)[None].astype("float32")
    p = model.predict(x, verbose=0)[0]
    label = CLASS_NAMES[int(np.argmax(p))]
    if label != "C3":
        wrong += 1
        print(name, "->", label, f"{p.max() * 100:.1f}%")

print("Total wrong:", wrong)