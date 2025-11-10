import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

DEFAULT_MODEL_PATH = Path("Best_Cattle_Breed.h5")
DEFAULT_CLASS_NAMES = Path("class_names.json")
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model(model_path: Path):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_names(class_names_path: Path):
    with open(class_names_path, "r") as f:
        names = json.load(f)
    return list(names)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_topk(model, class_names, image: Image.Image, top_k: int = 5):
    batch = preprocess_image(image)
    preds = model.predict(batch, verbose=0)[0]
    k = max(1, min(top_k, len(preds)))
    top_idx = preds.argsort()[-k:][::-1]
    return [(class_names[i], float(preds[i])) for i in top_idx], preds

st.set_page_config(page_title="Cattle Breed Classifier", page_icon="🐄", layout="wide")
st.title("🐄 Cattle Breed Classifier")
st.caption("Upload an image to get breed prediction with confidence scores.")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
    class_path = st.text_input("Class names path", str(DEFAULT_CLASS_NAMES))
    top_k = st.slider("Top‑K", 1, 10, 5)

    load_ok = True
    model = None
    class_names = None
    model_file = Path(model_path)
    class_file = Path(class_path)
    if not model_file.exists():
        st.error(f"Model not found: {model_file}")
        load_ok = False
    if not class_file.exists():
        st.error(f"class_names.json not found: {class_file}")
        load_ok = False
    if load_ok:
        model = load_model(model_file)
        class_names = load_class_names(class_file)
        st.success(f"Loaded model and {len(class_names)} classes.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

col_left, col_right = st.columns([1, 1])

if uploaded is not None and load_ok:
    image = Image.open(uploaded)
    with col_left:
        st.subheader("Preview")
        st.image(image, use_column_width=True)

    with col_right:
        st.subheader("Prediction")
        topk, raw = predict_topk(model, class_names, image, top_k=top_k)
        primary = topk[0]
        st.markdown(f"**Predicted:** {primary[0]}  \n**Confidence:** {primary[1]*100:.2f}%")

        st.markdown("Top‑K confidences:")
        labels = [name for name, _ in topk]
        scores = [p * 100 for _, p in topk]
        st.bar_chart({"confidence (%)": scores}, x=labels)

else:
    st.info("Upload an image to start. Make sure the model and class names files exist.")

