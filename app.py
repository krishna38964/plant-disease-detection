
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="LeafCare AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("model/plant_disease_model.keras")
CLASS_PATH = Path("model/class_names.txt")

# These thresholds help flag uncertain predictions.
# They do NOT guarantee detection of every unsupported leaf.
MIN_CONFIDENCE = 70.0
MIN_MARGIN = 15.0

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f3faf4 0%, #ffffff 55%, #e9f6ed 100%);
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #103d2b, #19734b);
    padding: 38px 35px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(20, 80, 50, 0.16);
}

.hero h1 {
    color: white;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    margin: 8px 0;
}

.hero p {
    color: #e0f3e5;
    font-size: 16px;
    line-height: 1.7;
}

.eyebrow {
    color: #b7f3c9;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.section-title {
    color: #164b32;
    font-size: 24px;
    font-weight: 800;
    margin: 20px 0 8px 0;
}

.info-card {
    background: white;
    padding: 22px;
    border: 1px solid #dcebe0;
    border-radius: 18px;
    min-height: 130px;
    box-shadow: 0 5px 18px rgba(25, 70, 40, 0.04);
}

.info-card h3 {
    color: #174b31;
    font-size: 17px;
    margin-bottom: 8px;
}

.info-card p {
    color: #617468;
    font-size: 14px;
    line-height: 1.6;
}

div.stButton > button {
    background: #187447;
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #105a35;
    color: white;
    border: none;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #9bc7a7;
    padding: 15px;
    border-radius: 15px;
}

.result-card {
    background: white;
    border: 1px solid #dcebe0;
    border-radius: 18px;
    padding: 24px;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #718176;
    font-size: 12px;
    margin-top: 40px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

def readable_name(name):
    return (
        name.replace("___", " - ")
            .replace("__", " ")
            .replace("_", " ")
    )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌿 LeafCare AI")
    st.caption("Plant health assistant")
    st.divider()

    st.markdown("### Supported plants")
    st.markdown("""
    - 🍅 Tomato
    - 🥔 Potato
    - 🫑 Bell Pepper
    """)

    st.divider()
    st.markdown("### How to use")
    st.write("1. Upload a clear leaf photo.")
    st.write("2. Click Detect Leaf.")
    st.write("3. Read the prediction or warning.")

    st.info(
        "This model was trained on specific plant-leaf classes. "
        "It may not reliably recognize other plants."
    )

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">SMART AGRICULTURE • AI TECHNOLOGY</div>
    <h1>🌿 LeafCare AI</h1>
    <p>
        Detect common plant leaf diseases with an AI-powered image model.
        Upload a leaf image to get a prediction and confidence score.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>📷 Easy Upload</h3>
        <p>Upload a clear JPG, JPEG or PNG image of a plant leaf.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 AI Prediction</h3>
        <p>The trained model compares your image with its learned classes.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>⚠️ Safety Warning</h3>
        <p>Uncertain predictions are flagged instead of presented as confirmed results.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">🔎 Check Your Plant Leaf</div>',
            unsafe_allow_html=True)
st.write("For best results, use a clear photo with the leaf in focus.")

# ---------------- CHECK MODEL FILES ----------------
if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error(
        "Model files were not found. Please run train_model.py first "
        "and make sure the model folder contains both required files."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of one leaf."
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("#### Your uploaded image")
            st.image(
                original_image,
                caption="Uploaded leaf",
                use_container_width=True
            )

        with right:
            st.markdown("#### Ready to analyse?")
            st.write(
                "Make sure the leaf is visible and the image is not too blurry."
            )

            detect_clicked = st.button(
                "🌱 Detect Leaf",
                use_container_width=True
            )

            st.caption(
                "Predictions are estimates and should not be treated "
                "as a confirmed plant diagnosis."
            )

        if detect_clicked:
            with st.spinner("Analysing the leaf image..."):
                resized_image = original_image.resize((128, 128))

                image_array = np.asarray(
                    resized_image, dtype=np.float32
                )
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]

                # Get the top two predicted classes
                top_indices = np.argsort(predictions)[::-1]
                best_index = int(top_indices[0])

                best_confidence = float(predictions[best_index]) * 100
                second_confidence = (
                    float(predictions[top_indices[1]]) * 100
                    if len(top_indices) > 1 else 0.0
                )

                margin = best_confidence - second_confidence
                predicted_class = class_names[best_index]
                display_class = readable_name(predicted_class)

            st.markdown("---")
            st.markdown(
                '<div class="section-title">📋 Analysis Result</div>',
                unsafe_allow_html=True
            )

            # This is a basic uncertainty check, not a true
            # unsupported-plant detector.
            if (
                best_confidence < MIN_CONFIDENCE
                or margin < MIN_MARGIN
            ):
                st.warning(
                    "⚠️ **Prediction uncertain!** We cannot confidently "
                    "identify this leaf. Please upload a clearer image "
                    "of a supported plant: Tomato, Potato or Bell Pepper."
                )

                st.metric("Top predicted class (unconfirmed)", display_class)
                st.metric("Model confidence", f"{best_confidence:.2f}%")

                st.info(
                    "No disease diagnosis is being confirmed because "
                    "the prediction did not pass the confidence checks."
                )

            else:
                st.success(
                    "The model found a confident match among its trained classes. "
                    "This does not guarantee that the uploaded plant is supported."
                )

                st.markdown('<div class="result-card">',
                            unsafe_allow_html=True)
                st.markdown("### 🌿 Predicted class")
                st.subheader(display_class)
                st.metric("Model confidence", f"{best_confidence:.2f}%")
                st.progress(
                    min(max(best_confidence / 100, 0.0), 1.0)
                )
                st.markdown('</div>', unsafe_allow_html=True)

                st.caption(
                    "Important: the model always chooses from its known classes. "
                    "An unsupported plant can still receive a high score, so "
                    "verify the plant identity before relying on this result."
                )

            with st.expander("View top predictions"):
                for idx in top_indices[:3]:
                    idx = int(idx)
                    st.write(
                        f"**{readable_name(class_names[idx])}** — "
                        f"{float(predictions[idx]) * 100:.2f}%"
                    )

    except Exception as error:
        st.error(
            "We could not read or analyse this image. "
            f"Please try another JPG or PNG image. Details: {error}"
        )

else:
    st.info(
        "👆 Upload a leaf image above to get started."
    )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    🌿 LeafCare AI · Plant Disease Detection Project<br>
    AI predictions are for educational purposes and are not a substitute
    for expert agricultural advice.
</div>
""", unsafe_allow_html=True)
