
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PlantCare AI | Smart Plant Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("model/plant_disease_model.keras")
CLASS_PATH = Path("model/class_names.txt")

# ---------------- PREMIUM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --green: #16845b;
    --dark: #123b2b;
    --light: #f2f8f3;
    --text: #19382a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #e0f3e5 0%, transparent 34%),
        linear-gradient(180deg, #f7fbf7 0%, #edf5ef 100%);
    color: var(--text);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #103d2c, #09291e);
}

[data-testid="stSidebar"] * {
    color: #f2fff6 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #ffffff30;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.7px;
}

.brand span {
    color: #86e4a7;
}

.sidebar-note {
    color: #d1e8d9;
    font-size: 0.91rem;
    line-height: 1.7;
}

.hero {
    background: linear-gradient(120deg, #103c2b 0%, #176e4b 65%, #23875b 100%);
    padding: 42px 40px;
    border-radius: 28px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 42px #174d2b20;
    margin-bottom: 25px;
}

.hero:after {
    content: "🌿";
    position: absolute;
    right: 35px;
    top: 18px;
    font-size: 105px;
    opacity: 0.15;
    transform: rotate(-18deg);
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.76rem;
    font-weight: 700;
    color: #b4f1c8;
}

.hero h1 {
    color: white !important;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2rem, 4.4vw, 3.4rem);
    line-height: 1.13;
    letter-spacing: -1.8px;
    margin: 14px 0;
    max-width: 720px;
}

.hero p {
    color: #e0f3e7;
    font-size: 1.05rem;
    line-height: 1.8;
    max-width: 680px;
    margin-bottom: 0;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #163e2c;
    margin: 12px 0 4px;
}

.section-subtitle {
    color: #6c8275;
    font-size: 0.96rem;
    margin-bottom: 20px;
}

.feature-card {
    background: rgba(255,255,255,0.9);
    border: 1px solid #dcebe0;
    border-radius: 20px;
    padding: 22px;
    min-height: 135px;
    box-shadow: 0 5px 20px #183d2710;
    margin-bottom: 12px;
}

.feature-icon {
    font-size: 1.8rem;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 1rem;
    font-weight: 800;
    color: #204a34;
    margin-bottom: 6px;
}

.feature-text {
    font-size: 0.88rem;
    line-height: 1.6;
    color: #718278;
}

.upload-panel {
    background: white;
    border: 1px solid #dcebe0;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 8px 28px #153c2410;
    margin-bottom: 18px;
}

.result-card {
    background: white;
    border: 1px solid #dcebe0;
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 8px 28px #153c2410;
    margin: 15px 0;
}

.result-label {
    color: #718278;
    font-size: 0.83rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-name {
    color: #174a31;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.25rem, 3vw, 2rem);
    font-weight: 800;
    margin: 10px 0;
    overflow-wrap: anywhere;
}

.pill {
    display: inline-block;
    background: #e0f5e6;
    color: #126b40;
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 0.82rem;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(110deg, #14774d, #229563);
    color: white;
    border: 0;
    border-radius: 13px;
    padding: 0.78rem 1.3rem;
    font-weight: 700;
    min-height: 3rem;
    transition: all 0.2s ease;
    box-shadow: 0 5px 15px #16845b25;
}

div.stButton > button:hover {
    background: linear-gradient(110deg, #105f3d, #16764d);
    color: white;
    border: 0;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: #f7fbf7;
    border: 1.5px dashed #a9cdb4;
    border-radius: 17px;
    padding: 15px;
}

[data-testid="stMetric"] {
    background: #f5faf5;
    border: 1px solid #dfede2;
    padding: 18px;
    border-radius: 16px;
}

hr {
    border-color: #dce9de;
}

.footer {
    text-align: center;
    color: #7b8c80;
    font-size: 0.86rem;
    padding: 25px 0 8px;
}

@media (max-width: 640px) {
    .hero {
        padding: 28px 22px;
        border-radius: 20px;
    }
    .hero:after {
        right: 10px;
        top: 12px;
        font-size: 55px;
    }
    .hero h1 {
        font-size: 2rem;
        letter-spacing: -1px;
    }
    .upload-panel, .result-card {
        padding: 17px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand">🌿 PlantCare <span>AI</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### Your Smart Plant Doctor")
    st.markdown(
        '<div class="sidebar-note">'
        'Check supported plant leaves with your AI model. '
        'Upload a clear photo to get a prediction.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("#### 🌱 Supported plants")
    st.markdown("- 🍅 Tomato")
    st.markdown("- 🥔 Potato")
    st.markdown("- 🫑 Bell pepper")
    st.markdown("---")
    st.markdown("#### 📋 How to use")
    st.markdown(
        '<div class="sidebar-note">'
        '1. Upload a leaf photo<br>'
        '2. Preview your image<br>'
        '3. Click Analyze Plant<br>'
        '4. Review the prediction'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.caption("AI predictions are informational and may be incorrect.")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ SMART AGRICULTURE • AI POWERED</div>
    <h1>Healthy plants start<br>with early detection.</h1>
    <p>
        Meet your digital plant doctor. Analyze a supported leaf image
        and explore an AI-generated disease prediction in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📸</div>
        <div class="feature-title">Easy image upload</div>
        <div class="feature-text">
            Upload a clear JPG or PNG leaf photograph.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">AI-powered analysis</div>
        <div class="feature-text">
            Get a prediction from your trained deep-learning model.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🌿</div>
        <div class="feature-title">Plant-focused results</div>
        <div class="feature-text">
            Designed for supported tomato, potato and pepper classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">🔬 Analyze your plant</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">'
    'For best results, use a well-lit photo showing one leaf clearly.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_image_checker():
    return tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True
    )

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

# ---------------- IMAGE VALIDATION ----------------
PLANT_KEYWORDS = [
    "plant", "leaf", "leaves", "flower", "tree",
    "vegetable", "bell_pepper", "cucumber", "broccoli",
    "cabbage", "cauliflower", "zucchini", "squash",
    "artichoke", "rapeseed", "daisy", "sunflower",
    "corn", "ear", "potato", "tomato", "greenhouse"
]

def looks_like_plant(image, image_checker):
    checker_image = image.resize((224, 224))
    array = np.asarray(checker_image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)

    predictions = image_checker.predict(array, verbose=0)
    top_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions, top=5
    )[0]

    return any(
        any(keyword in label.lower() for keyword in PLANT_KEYWORDS)
        for _, label, score in top_predictions
        if score >= 0.05
    )

# ---------------- UPLOAD PANEL ----------------
with st.container():
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📤 Choose your leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG."
    )

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Your uploaded plant image",
                use_container_width=True
            )
            st.caption(
                f"Image size: {image.width} × {image.height} pixels"
            )
        except Exception:
            st.error("This file could not be opened as an image.")
            image = None

    analyze = st.button(
        "🔍  Analyze Plant",
        type="primary",
        use_container_width=True,
        disabled=image is None
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if analyze and image is not None:
    try:
        with st.spinner("🌿 Preparing your plant analysis..."):
            model = load_disease_model()
            image_checker = load_image_checker()
            class_names = load_class_names()

        with st.spinner("🧠 AI is analyzing your image..."):
            if not looks_like_plant(image, image_checker):
                st.error(
                    "🚫 This image does not look like a supported plant "
                    "image. Please upload a clear leaf photo instead."
                )
                st.info(
                    "Tip: Photograph one tomato, potato or bell pepper "
                    "leaf in good lighting."
                )
                st.stop()

            model_height = model.input_shape[1]
            model_width = model.input_shape[2]

            resized = image.resize((model_width, model_height))
            image_array = np.asarray(resized, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]

            if len(class_names) != len(predictions):
                st.error(
                    "The number of class names does not match the model "
                    "outputs. Please check model/class_names.txt."
                )
                st.stop()

            best_index = int(np.argmax(predictions))
            confidence = float(predictions[best_index])
            disease_name = class_names[best_index]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">📋 Your analysis results</div>',
            unsafe_allow_html=True
        )

        if confidence < 0.70:
            st.warning(
                "The prediction is uncertain. Try a clearer leaf photo "
                "with good lighting and less background clutter."
            )
        else:
            st.success("Plant image check passed.")

        r1, r2 = st.columns([1.5, 1])

        with r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">AI prediction</div>
                <div class="result-name">{disease_name}</div>
                <span class="pill">
                    {"Needs review" if confidence < 0.70 else "Analysis complete"}
                </span>
                <p style="color:#718278;line-height:1.7;margin-top:18px;">
                    This result is generated by your trained plant
                    disease model. Check the leaf and symptoms before
                    making treatment decisions.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Model confidence")
            st.metric("Highest model score", f"{confidence * 100:.2f}%")
            st.progress(max(0.0, min(1.0, confidence)))
            st.caption(
                "Confidence is not the percentage of the leaf that is "
                "damaged, and it does not guarantee a correct result."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("💡 Tips for better results"):
            st.markdown("""
            - Use a clear, in-focus image.
            - Keep the leaf well lit.
            - Avoid covering the leaf with your hand.
            - Use a leaf from a supported plant.
            - Compare the result with visible symptoms and expert advice.
            """)

    except Exception as error:
        st.error(
            "The image could not be analyzed. Please try again or "
            "check the Streamlit app logs."
        )
        # Technical details are available for troubleshooting.
        with st.expander("Technical details"):
            st.code(str(error))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="footer">
    🌿 <b>PlantCare AI</b> · Smart plant disease prediction<br>
    Built with Streamlit & TensorFlow<br>
    <span>For educational purposes. AI predictions may be incorrect.</span>
</div>
""", unsafe_allow_html=True)
