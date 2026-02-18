"""
Enhanced Cattle Breed Classifier with AI Features
Includes: Classification, AI Chatbot, Batch Processing, Similarity Search
"""

import json
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
from keras_compat import keras
from PIL import Image

# Import AI modules
from ai_chatbot import CattleBreedChatbot
from batch_processor import BatchProcessor
from similarity_search import SimilaritySearch

DEFAULT_MODEL_PATH = Path("Best_Cattle_Breed.h5")
DEFAULT_CLASS_NAMES = Path("class_names.json")
IMAGE_SIZE = (224, 224)

# Page configuration
st.set_page_config(
    page_title="Cattle Breed Classifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'similarity_searcher' not in st.session_state:
    st.session_state.similarity_searcher = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
    class_path = st.text_input("Class names path", str(DEFAULT_CLASS_NAMES))
    
    st.divider()
    st.subheader("🎨 Appearance")
    accent = st.color_picker("Accent color", "#4F46E5", key="accent_color")
    accent2 = st.color_picker("Accent secondary", "#22C55E", key="accent_color2")
    dark_mode = st.toggle("Dark mode", value=True, key="dark_mode")
    
    # CSS styling
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
    
    # Load resources
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
        st.success(f"✅ Loaded model and {len(class_names)} classes.")
        
        # Initialize chatbot
        if st.session_state.chatbot is None:
            st.session_state.chatbot = CattleBreedChatbot(class_file)

# Main content
st.markdown(
    """
    <div class="app-hero">
        <h1 style="margin:0;">🤖 Cattle Breed Classifier</h1>
        <div class="muted" style="margin-top:6px;">
            Advanced classification with AI chatbot, batch processing, and similarity search.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs for different features
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Classification", 
    "💬 AI Chatbot", 
    "📦 Batch Processing",
    "🔎 Similarity Search"
])

# Tab 1: Classification
with tab1:
    st.subheader("Image Classification")
    top_k = st.slider("Top‑K predictions", 1, 10, 5, key="topk_classification")
    
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="classify_upload")
    sample = find_sample_image()
    if sample is not None and uploaded is None:
        if st.button("Try a sample image"):
            with open(sample, "rb") as f:
                uploaded = BytesIO(f.read())
    
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
        st.info("Upload an image to start classification.")

# Tab 2: AI Chatbot
with tab2:
    st.subheader("💬 AI Chatbot - Ask About Cattle Breeds")
    st.caption("Ask questions about cattle breeds, their characteristics, origins, and more!")
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about cattle breeds..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get chatbot response
        if st.session_state.chatbot:
            response = st.session_state.chatbot.respond(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            st.error("Chatbot not initialized. Please check model and class names files.")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Tab 3: Batch Processing
with tab3:
    st.subheader("📦 Batch Processing - Process Multiple Images")
    st.caption("Upload multiple images or select a directory to process in batch")
    
    batch_mode = st.radio("Processing mode", ["Upload Multiple Files", "Process Directory"], horizontal=True)
    
    if batch_mode == "Upload Multiple Files":
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files and load_ok:
            if st.button("Process Batch", type="primary"):
                with st.spinner("Processing images..."):
                    try:
                        processor = BatchProcessor(model_file, class_file)
                        
                        # Save uploaded files temporarily
                        temp_dir = Path("temp_batch")
                        temp_dir.mkdir(exist_ok=True)
                        temp_paths = []
                        
                        for uploaded_file in uploaded_files:
                            temp_path = temp_dir / uploaded_file.name
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)
                        
                        # Process
                        results = processor.predict_batch(temp_paths, top_k=5)
                        
                        # Display results
                        st.success(f"✅ Processed {len(results)} images")
                        
                        # Create DataFrame
                        import pandas as pd
                        df_data = []
                        for result in results:
                            df_data.append({
                                'Image': result['image_name'],
                                'Predicted Breed': result['top_breed'],
                                'Confidence (%)': f"{result['top_confidence']*100:.2f}"
                            })
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Statistics
                        stats = processor.get_statistics(results)
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Total Images", stats['total_images'])
                        with col_stat2:
                            st.metric("Unique Breeds", stats['unique_breeds'])
                        with col_stat3:
                            st.metric("Avg Confidence", f"{stats['average_confidence']:.2f}%")
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv,
                            file_name="batch_results.csv",
                            mime="text/csv"
                        )
                        
                        # Cleanup
                        for path in temp_paths:
                            path.unlink()
                        temp_dir.rmdir()
                        
                    except Exception as e:
                        st.error(f"Error processing batch: {e}")
    else:
        st.info("Directory processing coming soon! Use 'Upload Multiple Files' for now.")

# Tab 4: Similarity Search
with tab4:
    st.subheader("🔎 Similarity Search - Find Similar Cattle Images")
    st.caption("Find visually similar cattle images from the database")
    
    uploaded_similarity = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="similarity_upload")
    top_k_similar = st.slider("Number of similar images", 1, 10, 5, key="topk_similarity")
    
    if uploaded_similarity is not None and load_ok:
        query_image = Image.open(uploaded_similarity)
        st.image(query_image, caption="Query Image", width=300)
        
        if st.button("Find Similar Images", type="primary"):
            with st.spinner("Building database and searching..."):
                try:
                    # Initialize searcher
                    if st.session_state.similarity_searcher is None:
                        st.session_state.similarity_searcher = SimilaritySearch(
                            model_file, class_file, Path("data")
                        )
                        st.session_state.similarity_searcher.build_database(max_images_per_breed=20)
                    
                    # Save query image temporarily
                    temp_query = Path("temp_query.jpg")
                    query_image.save(temp_query)
                    
                    # Find similar
                    results = st.session_state.similarity_searcher.find_similar(
                        temp_query, top_k=top_k_similar
                    )
                    
                    # Display results
                    st.success(f"✅ Found {len(results)} similar images")
                    
                    cols = st.columns(min(len(results), 5))
                    for i, result in enumerate(results):
                        with cols[i % len(cols)]:
                            try:
                                img = Image.open(result['image_path'])
                                st.image(img, caption=f"{result['breed']}\n{result['similarity']*100:.1f}% similar", use_container_width=True)
                            except:
                                st.write(f"Could not load: {result['image_path'].name}")
                    
                    # Cleanup
                    temp_query.unlink()
                    
                except Exception as e:
                    st.error(f"Error in similarity search: {e}")
                    st.info("Make sure the 'data' directory exists with cattle breed images.")
    else:
        st.info("Upload a query image to find similar cattle images.")

# Footer
st.divider()
st.caption("🤖 Cattle Breed Classification System | Built with TensorFlow, Streamlit, and Advanced AI Features")

