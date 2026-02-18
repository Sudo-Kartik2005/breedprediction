import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from keras_compat import keras
from PIL import Image

DEFAULT_MODEL_PATH = Path("Best_Cattle_Breed.h5")
DEFAULT_CLASS_NAMES = Path("class_names.json")
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model(model_path: Path):
    return keras.models.load_model(model_path)

@st.cache_data
def load_class_names(class_names_path: Path):
    with open(class_names_path, "r") as f:
        names = json.load(f)
    return list(names)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = keras.applications.efficientnet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_topk(model, class_names, image: Image.Image, top_k: int = 5):
    batch = preprocess_image(image)
    preds = model.predict(batch, verbose=0)[0]
    k = max(1, min(top_k, len(preds)))
    top_idx = preds.argsort()[-k:][::-1]
    return [(class_names[i], float(preds[i])) for i in top_idx], preds

def find_sample_image() -> Path | None:
    data_dir = Path("data")
    if not data_dir.exists() or not data_dir.is_dir():
        return None
    for sub in sorted(data_dir.iterdir()):
        if sub.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                imgs = list(sub.glob(ext))
                if imgs:
                    return imgs[0]
    return None

st.set_page_config(page_title="Cattle Breed Classifier", page_icon="🐄", layout="wide")
st.markdown(
    """
    <div class="app-hero">
        <h2 style="margin:0;">🐄 Cattle Breed Classifier</h2>
        <div class="muted" style="margin-top:6px;">
            Upload a cattle image to get the predicted breed with confidence and a top‑K chart.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
    class_path = st.text_input("Class names path", str(DEFAULT_CLASS_NAMES))
    top_k = st.slider("Top‑K", 1, 10, 5)

    st.subheader("Appearance")
    accent = st.color_picker("Accent color", "#4F46E5", key="accent_color")
    accent2 = st.color_picker("Accent secondary", "#22C55E", key="accent_color2")
    dark_mode = st.toggle("Dark mode", value=True, key="dark_mode")

    # Inject dynamic CSS based on appearance controls
    bg_primary = "#0B1217" if dark_mode else "#FFFFFF"
    bg_secondary = "#111827" if dark_mode else "#F3F4F6"
    text_color = "#E5E7EB" if dark_mode else "#0B1217"
    card_bg = "rgba(255,255,255,0.04)" if dark_mode else "#FFFFFF"
    border_color = "rgba(148,163,184,0.15)" if dark_mode else "rgba(0,0,0,0.06)"

    st.markdown(
        f"""
        <style>
            :root {{
                --brand: {accent};
                --brand-2: {accent2};
                --card-bg: {card_bg};
                --text-muted: {"#9AA4B2" if dark_mode else "#6B7280"};
            }}
            .app-hero {{
                padding: 16px 20px;
                border-radius: 16px;
                background: linear-gradient(135deg, color-mix(in oklab, var(--brand) 20%, transparent), color-mix(in oklab, var(--brand-2) 18%, transparent));
                border: 1px solid {border_color};
                margin-bottom: 8px;
            }}
            .pred-card {{
                padding: 16px;
                border-radius: 14px;
                background: var(--card-bg);
                border: 1px solid {border_color};
            }}
            .pill {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                background: color-mix(in oklab, var(--brand) 18%, transparent);
                color: {text_color};
                border: 1px solid color-mix(in oklab, var(--brand) 40%, transparent);
                font-weight: 600;
                letter-spacing: .2px;
            }}
            .muted {{
                color: var(--text-muted);
            }}
            section[data-testid="stSidebar"] {{
                color: {text_color};
            }}
            footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    st.divider()
    st.caption("Tips")
    st.write(
        "- Use clear side-view images.\n"
        "- Higher resolution yields better results.\n"
        "- Lighting and full body help accuracy."
    )
    sample = find_sample_image()
    if sample is not None:
        if st.button("Try a sample image"):
            with open(sample, "rb") as f:
                st.session_state["sample_bytes"] = f.read()
    else:
        st.caption("No sample image found in `data/`.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if "sample_bytes" in st.session_state and uploaded is None:
    from io import BytesIO
    uploaded = BytesIO(st.session_state["sample_bytes"])

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
        st.markdown(f'<span class="pill">Predicted: {primary[0]}</span>', unsafe_allow_html=True)
        st.progress(min(max(primary[1], 0.0), 1.0), text=f"Confidence: {primary[1]*100:.2f}%")

        st.markdown("Top‑K confidences:")
        labels = [name for name, _ in topk]
        scores = [p * 100 for _, p in topk]
        try:
            import altair as alt
            import pandas as pd
            df = pd.DataFrame({"Breed": labels, "Confidence": scores})
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    x=alt.X("Confidence:Q", scale=alt.Scale(domain=[0, 100]), title="Confidence (%)"),
                    y=alt.Y("Breed:N", sort="-x", title=None),
                    color=alt.value(st.session_state.get("accent_color", "#4F46E5")),
                    tooltip=["Breed", alt.Tooltip("Confidence:Q", format=".2f")]
                )
                .properties(height=max(180, 24 * len(df)))
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.bar_chart({"confidence (%)": scores}, x=labels)

else:
    st.info("Upload an image to start. Make sure the model and class names files exist.")

