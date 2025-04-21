import os
import urllib.request
import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# File setup
MODEL_PATH = "model.h5"
MODEL_ID = "122wvaD-tM23HNit-oQMsGRfQpuTfB1Dk"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

model = load_model(MODEL_PATH)

#Class names (index 0 = NORMAL, index 1 = PNEUMONIA)
class_names = ['NORMAL', 'PNEUMONIA']

#Streamlit page setup
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("""
         Welcome to PaediaScan
         Your AI-Assisted X-ray Review for Paediatric Pneumonia
         """
         )
uploaded_file = st.file_uploader("Please upload a chest x-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    #Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    # Make prediction
    prediction = model.predict(img_array)
    pneumonia_confidence = float(prediction[0][1])

    #Applying threshold logic
    threshold = 0.685  
    if pneumonia_confidence > threshold:
        predicted_class = "PNEUMONIA"
        confidence = pneumonia_confidence
    else:
        predicted_class = "NORMAL"
        confidence = 1 - pneumonia_confidence

    #Display prediction results
    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"This image most likely represents a {predicted_class} chest x-ray")
