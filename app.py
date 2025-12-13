import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("saved_model/mobilenet_model.h5")

model = load_model()


CLASS_NAMES = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]

IMG_SIZE = (128, 128)

# UI
st.title("Fish Species Recognition")
st.write("Upload a fish image and the model will predict its species.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Preprocess
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Top-3 predictions
        st.subheader("Top 3 Predictions")
        top_3_idx = predictions.argsort()[-3:][::-1]

        for idx in top_3_idx:
            st.write(f"{CLASS_NAMES[idx]}: {predictions[idx]*100:.2f}%")
