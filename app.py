# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import warnings

# Suppress only deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# %%
import streamlit as st
import numpy as np
import cv2
import joblib

from tensorflow.keras.models import load_model


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Car Classification CNN",
    page_icon="🚗",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_cnn_model():
    return load_model("model.keras")


@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")


model = load_cnn_model()
label_encoder = load_label_encoder()


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🚗 Car Classification System")

st.write(
    "Upload a car image and the CNN model will classify it "
    "as Suzuki, Toyota, or Others."
)


# -----------------------------
# Image Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png", "webp"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Check image
    if image is None:
        st.error("Unable to read the uploaded image.")
        st.stop()

    # Convert BGR → RGB for display
    display_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        display_image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------

    resized_image = cv2.resize(
        image,
        (150, 150)
    )

    resized_image = resized_image.astype(
        "float32"
    ) / 255.0

    # Add batch dimension
    input_image = np.expand_dims(
        resized_image,
        axis=0
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(
        input_image,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction,
        axis=1
    )[0]

    confidence = np.max(prediction) * 100

    # -----------------------------
    # Threshold
    # -----------------------------

    THRESHOLD = 70

    if confidence < THRESHOLD:

        predicted_label = "Image Not Recognized"

    else:

        predicted_label = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

    # -----------------------------
    # Display Result
    # -----------------------------

    st.subheader("Prediction")

    st.success(
        f"Prediction: {predicted_label}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

# %%
import os

size_mb = os.path.getsize("model.keras") / (1024 * 1024)

print(f"Model size: {size_mb:.2f} MB")

# %%
