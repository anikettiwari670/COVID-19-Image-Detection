import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    page_icon="🫁",
    layout="centered"
)

# ── Constants (must match notebook) ───────────────────────────────────────────
IMG_SIZE = 128
CLASSES  = ["Covid", "Normal", "Viral Pneumonia"]
CLASS_COLORS = {"Covid": "🔴", "Normal": "🟢", "Viral Pneumonia": "🟡"}

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_best_model():
    return load_model("COVID-19_best_model.keras")

model = load_best_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🫁 About")
    st.info(
        "Upload a chest X-ray image and this app will classify it as "
        "**Covid**, **Normal**, or **Viral Pneumonia** using a trained CNN model."
    )
    st.divider()
    st.markdown("**Model:** Basic CNN (best performer)")
    st.markdown("**Input size:** 128 × 128 px")
    st.markdown("**Classes:** Covid · Normal · Viral Pneumonia")
    st.markdown("**Preprocessing:** BGR→RGB, resize, normalize (/255)")
    st.divider()

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🫁 COVID-19 X-Ray Classifier")
st.markdown("Upload a chest X-ray image to detect **Covid**, **Normal**, or **Viral Pneumonia**.")
st.divider()

# ── File uploader ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Accepted formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:

    # Show uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded X-Ray")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    # ── Preprocess (same as notebook) ─────────────────────────────────────────
    img_array = np.array(image)                              # PIL → numpy array(RGB)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # RGB → BGR (OpenCV)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)) # resize to 128x128
    img_normalized = img_resized / 255.0                    # normalize to [0,1]
    img_input = np.expand_dims(img_normalized, axis=0)      # add batch dim → (1,128,128,3)

    # ── Predict ────────────────────────────────────────────────────────────────
    with col2:
        st.subheader("Prediction")
        with st.spinner("Analyzing X-ray..."):
            predictions = model.predict(img_input)[0]       # shape: (3,)

        predicted_idx   = np.argmax(predictions)
        predicted_class = CLASSES[predicted_idx]
        confidence      = predictions[predicted_idx] * 100

        emoji = CLASS_COLORS[predicted_class]

        if predicted_class == "Covid":
            st.error(f"{emoji} **{predicted_class}**", icon="🚨")
        elif predicted_class == "Normal":
            st.success(f"{emoji} **{predicted_class}**", icon="✅")
        else:
            st.warning(f"{emoji} **{predicted_class}**", icon="⚠️")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(float(predictions[predicted_idx]))

    # ── All class probabilities ────────────────────────────────────────────────
    st.divider()
    st.subheader("Class probabilities")
    for i, cls in enumerate(CLASSES):
        prob = predictions[i] * 100
        st.markdown(f"**{CLASS_COLORS[cls]} {cls}**")
        st.progress(float(predictions[i]), text=f"{prob:.2f}%")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ This tool is for educational purposes only and is not a substitute "
        "for professional medical diagnosis."
    )

else:
    # Placeholder when no image uploaded
    st.info("👆 Upload a chest X-ray image above to get started.")
