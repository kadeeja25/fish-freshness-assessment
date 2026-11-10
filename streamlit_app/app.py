import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CLASS_NAMES = ["C1", "C2", "C3"]
LABELS = {"C1": "Fresh", "C2": "Moderately fresh", "C3": "Spoiled"}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fish_freshness_model.keras",
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

st.title("Fish Freshness Checker")
model = load_model()
file = st.file_uploader("Upload a fish photo", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width="stretch")
    if st.button("Check freshness"):
        x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
        probs = model.predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        conf = float(probs[i]) * 100
        st.subheader(LABELS[CLASS_NAMES[i]] + " (" + str(round(conf, 2)) + "%)")
        if conf < 60:
            st.warning("Low confidence. Try a clearer, well-lit photo.")
        for n, p in zip(CLASS_NAMES, probs):
            st.write(LABELS[n] + ": " + str(round(float(p) * 100, 2)) + "%")
            st.progress(float(p))