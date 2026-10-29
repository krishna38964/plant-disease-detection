
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

MODEL_PATH = Path("model/plant_disease_model.keras")
CLASS_PATH = Path("model/class_names.txt")

# Load the disease model
@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load an independent, pretrained image classifier
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

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a clear leaf image of tomato, potato, "
    "or bell pepper to check for supported diseases."
)

try:
    model = load_disease_model()
    image_checker = load_image_checker()
    class_names = load_class_names()
except Exception as error:
    st.error("The models could not be loaded. Please check the app logs.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# ImageNet labels that can indicate plants or plant parts.
PLANT_KEYWORDS = [
    "plant", "leaf", "leaves", "flower", "tree",
    "vegetable", "bell_pepper", "cucumber", "broccoli",
    "cabbage", "cauliflower", "zucchini", "squash",
    "artichoke", "rapeseed", "daisy", "sunflower",
    "corn", "ear", "potato", "tomato", "greenhouse"
]

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Detect Disease", type="primary"):
            with st.spinner("Checking image..."):
                # First check: does a general image classifier
                # recognise a likely plant or plant part?
                checker_image = image.resize((224, 224))
                checker_array = np.asarray(
                    checker_image, dtype=np.float32
                )
                checker_array = np.expand_dims(checker_array, axis=0)
                checker_array = (
                    tf.keras.applications.mobilenet_v2.preprocess_input(
                        checker_array.copy()
                    )
                )

                checker_predictions = image_checker.predict(
                    checker_array, verbose=0
                )
                top_predictions = (
                    tf.keras.applications.mobilenet_v2.decode_predictions(
                        checker_predictions, top=5
                    )[0]
                )

                likely_plant = any(
                    any(keyword in label.lower()
                        for keyword in PLANT_KEYWORDS)
                    for _, label, score in top_predictions
                    if score >= 0.05
                )

                if not likely_plant:
                    st.error(
                        "❌ Invalid image! Please upload a clear "
                        "plant leaf image, not a person or unrelated object."
                    )
                    st.stop()

                # Second check: run the trained disease model
                model_height = model.input_shape[1]
                model_width = model.input_shape[2]

                resized = image.resize((model_width, model_height))
                image_array = np.asarray(resized, dtype=np.float32)

                # PlantVillage models are commonly trained on pixel
                # values in the 0-1 range.
                image_array = image_array / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]
                best_index = int(np.argmax(predictions))
                confidence = float(predictions[best_index])
                disease_name = class_names[best_index]

            # Low confidence is uncertain, not a confirmed disease.
            if confidence < 0.70:
                st.warning(
                    "⚠️ Prediction uncertain. Please upload a clear "
                    "leaf image and try again."
                )
                st.write(f"Highest model score: {confidence * 100:.2f}%")
            else:
                st.success("Plant image check passed.")
                st.subheader("Prediction")
                st.write(f"**Result:** {disease_name}")
                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )
                st.caption(
                    "This is the model's confidence score, not a "
                    "measured percentage of leaf damage. Results can "
                    "still be incorrect."
                )

    except Exception:
        st.error(
            "Could not process this image. Please upload a valid "
            "JPG or PNG image and try again."
        )
