import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "fish_freshness_model.keras"
IMG_PATH =r"C:\Users\Hawwa Shadiya\Desktop\fish freshness\Freshness Fish Dataset\C1\C:\Users\Hawwa Shadiya\Desktop\fish freshness\Freshness Fish Dataset\C1\WIN_20241125_14_50_20_Pro.jpg"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["C1", "C2", "C3"]

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False,
    compile=False,
)

# the nested MobileNetV2 inside your model
base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
conv_model = tf.keras.Model(
    base.input, [base.get_layer("out_relu").output, base.output]
)

# load image (raw 0-255 pixels, the model preprocesses internally)
img = Image.open(IMG_PATH).convert("RGB").resize(IMG_SIZE)
orig = np.array(img)
x = tf.convert_to_tensor(orig[np.newaxis, ...], dtype=tf.float32)

# forward pass layer by layer so we can grab the conv output
with tf.GradientTape() as tape:
    out = x
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer is base:
            conv_out, out = conv_model(out)
        else:
            out = layer(out)
    class_idx = int(tf.argmax(out[0]))
    score = out[:, class_idx]

grads = tape.gradient(score, conv_out)
weights = tf.reduce_mean(grads, axis=(1, 2))               # importance per channel
cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
cam = tf.nn.relu(cam).numpy()
cam = cam / (cam.max() + 1e-8)

# resize heatmap and overlay
heat = np.array(Image.fromarray(np.uint8(255 * cam)).resize(IMG_SIZE, Image.BILINEAR)) / 255.0
colored = plt.cm.jet(heat)[..., :3]
overlay = 0.6 * (orig / 255.0) + 0.4 * colored

conf = float(out[0][class_idx]) * 100
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(orig);    ax[0].set_title("Original")
ax[1].imshow(heat, cmap="jet"); ax[1].set_title("Grad-CAM heatmap")
ax[2].imshow(overlay); ax[2].set_title(f"Predicted: {CLASS_NAMES[class_idx]} ({conf:.1f}%)")
for a in ax: a.axis("off")
plt.tight_layout()
plt.savefig("gradcam_C1_img2.png", dpi=200)
plt.show()