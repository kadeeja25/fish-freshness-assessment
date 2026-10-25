import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "fish_freshness_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["C1", "C2", "C3"]

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False,
    compile=False,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img = img.convert("RGB").resize(IMG_SIZE)
    except Exception:
        raise HTTPException(400, "Invalid image file")
    x = np.expand_dims(np.array(img, dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[i],
        "confidence": round(float(probs[i]) * 100, 2),
        "probabilities": {
            n: round(float(p) * 100, 2) for n, p in zip(CLASS_NAMES, probs)
        },
    }
