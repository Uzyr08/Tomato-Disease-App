import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

# === Load model and disease data ===
model = load_model("model/Model9.h5")

with open("database/disease_info.json", "r") as f:
    disease_data = json.load(f)

# === Streamlit Page Config ===
st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🍅",
    layout="centered"
)

# === Custom Dark Green + Pale Yellow Theme ===
st.markdown("""
    <style>
        /* Overall background */
        .stApp {
            background-color: #204E24; /* Deep dark green */
            color: #FAF3C0; /* Pale yellow text */
        }

        html, body, [class*="st-"], div, p, span, label {
            color: #FAF3C0 !important;
            font-family: "Segoe UI", sans-serif !important;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #FFD95A !important; /* Soft golden yellow */
            font-weight: 700 !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #388E3C !important; /* Bright green */
            color: #FAF3C0 !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: bold !important;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #2E7D32 !important;
            transform: scale(1.02);
        }

        /* File uploader text */
        .stFileUploader label {
            color: #FFD95A !important;
            font-weight: 600 !important;
        }

        /* Result Box */
        .result-box {
            background-color: #2E7D32 !important;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 0 15px rgba(0,0,0,0.3);
            color: #FAF3C0 !important;
        }

        .probability {
            color: #FFF176 !important; /* Lighter yellow for confidence */
            font-weight: bold;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 0.9em;
            color: #DCE775 !important;
        }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("<h1 style='text-align:center;'>🍅 Tomato Disease Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a tomato leaf image to detect the disease and get treatment advice.</p>", unsafe_allow_html=True)

# === Upload Section ===
uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Show image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        prediction = model.predict(img_array)
        predicted_class = str(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        # Results
        if predicted_class in disease_data:
            result = disease_data[predicted_class]

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"### 🦠 Disease: {result['disease']}")
            st.markdown(f"**🔍 Description:** {result['description']}")
            st.markdown(f"**💊 Cure:** {result['cure']}")
            st.markdown(f"**📊 Confidence:** <span class='probability'>{confidence:.2f}%</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No treatment info found for this class.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

# === Footer ===
st.markdown("<div class='footer'>Developed by Uzair | Powered by Streamlit & TensorFlow</div>", unsafe_allow_html=True)
