import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "saved_models/emotion_cnn.h5"
EMOTIONS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def predict_emotion(image: Image.Image):
    # Convert to grayscale
    gray = image.convert("L")

    # Resize to 48x48 for CNN
    gray = gray.resize((48, 48))

    # Convert to NumPy array and normalize
    img_array = np.array(gray).astype("float32") / 255.0

    # Reshape for CNN input
    img_array = np.expand_dims(img_array, axis=-1)  # (48,48,1)
    img_array = np.expand_dims(img_array, axis=0)   # (1,48,48,1)

    # Predict
    predictions = model.predict(img_array)
    emotion_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return EMOTIONS[emotion_index], confidence

# Streamlit UI
st.title("😊 Emotion Detection App ")
st.write("Upload an image and the CNN model will predict the emotion.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image with PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict Emotion"):
        emotion, confidence = predict_emotion(image)
        st.subheader(f"Predicted Emotion: **{emotion}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
