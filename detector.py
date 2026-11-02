
import os
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/soybean_rust_mobilenet.keras"
IMG_SIZE = 224


@st.cache_resource
def load_disease_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


def predict_disease(pil_image):

    model = load_disease_model()

    # Convert and resize image
    img = pil_image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert image into NumPy array
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing
    img_array = preprocess_input(img_array)

    # Model prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    probability = float(prediction[0][0])

    # Determine class
    if probability >= 0.5:

        predicted_class = "Soybean Rust"
        confidence = probability * 100

    else:

        predicted_class = "Healthy Soybean Leaf"
        confidence = (1 - probability) * 100

    return predicted_class, confidence
