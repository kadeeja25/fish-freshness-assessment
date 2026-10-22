import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "fish_freshness_model.keras"
IMG_PATH=r"C:\Users\Hawwa Shadiya\Desktop\fish freshness\Freshness Fish Dataset\C1\WIN_20241125_14_49_56_Pro.jpg"
OUT_NAME = "crop_test_C1_img2.png"   # <-- change per image
IMG_SIZE = (224, 224)
CLASS_NAMES = ["C1", "C2", "C3"]

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False,
    compile=False,
)

def predict(arr):
    p = model.predict(arr[None].astype("float32"), verbose=0)[0]
    i = int(np.argmax(p))
    return CLASS_NAMES[i], p[i] * 100

img = np.array(Image.open(IMG_PATH).convert("RGB").resize(IMG_SIZE))
gray = img.mean(axis=2)

# find the bright fish, ignoring the metal strips at the left/right edges
center = gray[:, 30:-30]
thr = center.mean() + 2 * center.std()
mask = gray > thr
mask[:, :30] = False
mask[:, -30:] = False

if not mask.any():
    raise SystemExit("Fish not found. Try lowering the threshold.")

ys, xs = np.where(mask)
pad = 12
y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, IMG_SIZE[1])
x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, IMG_SIZE[0])

fish_only = np.zeros_like(img)
fish_only[y1:y2, x1:x2] = img[y1:y2, x1:x2]

bg_only = img.copy()
bg_only[y1:y2, x1:x2] = 0

results = [
    ("Original", img),
    ("Fish only", fish_only),
    ("Background only", bg_only),
]

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, (name, arr) in zip(ax, results):
    label, conf = predict(arr)
    a.imshow(arr)
    a.set_title(f"{name}\n{label} ({conf:.1f}%)")
    a.axis("off")
    print(name, "->", label, f"{conf:.1f}%")

plt.tight_layout()
plt.savefig(OUT_NAME, dpi=200)
plt.show()