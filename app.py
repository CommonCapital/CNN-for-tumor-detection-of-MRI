import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# --- Constants ---
MODEL_JSON_PATH = "classifier-resnet-model.json"
MODEL_WEIGHTS_PATH = "classifier-resnet-weights.hdf5"
MODEL_WEIGHTS_URL = "https://drive.google.com/uc?id=1DQK41XefRLrFqSXhiH3tIC1oyWub5ucy"

# --- Set up page title ---
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection using CNN")
st.markdown(
    "This app uses a Convolutional Neural Network (CNN) to detect brain tumors from MRI scans.\n\n"
    "**⚠️ Warning:** Only upload brain MRI scans sized (256, 256, 3), or results may be inaccurate."
)

# --- Load model architecture and weights ---
@st.cache_resource
def load_model():
    # Download weights file if missing
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        with st.spinner("📦 Downloading model weights..."):
            gdown.download(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH, quiet=False)

    # Load model architecture
    with open(MODEL_JSON_PATH, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model

model = load_model()

# --- Class labels ---
class_names = ["No Tumor", "Tumor"]

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Upload a brain MRI scan image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    # Display prediction
    st.subheader("🧪 Prediction Result:")
    if predicted_class == 1:
        st.success(f"✅ Tumor Detected with {confidence*100:.2f}% confidence.")
    else:
        st.error(f"❌ No Tumor Detected with {confidence*100:.2f}% confidence.")

    st.write("🔍 Prediction Probabilities:", {class_names[i]: f"{prob:.4f}" for i, prob in enumerate(prediction[0])})

# --- Contributors ---
st.markdown("---")
st.markdown("👨‍🔬 This project was built to **democratize AI for healthcare** by:")
st.markdown("- **Nursan Omarov**, 17-year-old AI engineer")
st.markdown("- **Alimzhan Tokushev**, 19-year-old ML engineer")
st.caption("📡 Empowering medical diagnostics through open technology.")