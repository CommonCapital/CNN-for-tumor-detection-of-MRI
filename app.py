import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# --- Set up page title ---
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection using CNN")
st.markdown("This app uses a Convolutional Neural Network (CNN) to detect brain tumors from MRI scans."
""
"Warning you should upload only brain MRI scans with size shape of (256,256,3) or it will not work properly!!!")

# --- Load model architecture and weights ---
@st.cache_resource
def load_model():
    with open("classifier-resnet-model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("classifier-resnet-weights.hdf5")
    return model

model = load_model()

# --- Class labels ---
class_names = ["No Tumor", "Tumor"]

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Upload a brain MRI scan image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
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

    # Optional: show raw prediction values
    st.write("🔍 Prediction Probabilities:", {class_names[i]: f"{prob:.4f}" for i, prob in enumerate(prediction[0])})

# --- Contributors ---
st.markdown("---")
st.markdown("👨‍🔬 This project was built to **democratize AI for healthcare** by:")
st.markdown("- **Nursan Omarov**, 17-year-old AI engineer")
st.markdown("- **Alimzhan Tokushev**, 19-year-old ML engineer")
st.caption("📡 Empowering medical diagnostics through open technology.")