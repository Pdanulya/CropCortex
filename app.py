import streamlit as st

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

# Load your model
from tensorflow.keras.models import load_model

model = load_model("plant_disease_final (1).keras")

# Image upload
from PIL import Image

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Preprocess image
import numpy as np

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class_names = [
 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy',
 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot', 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
 'Raspberry___healthy', 'Soybean___healthy',
 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Prediction logic
if uploaded_file is not None:
    processed_img = preprocess_image(image)

    prediction = model.predict(processed_img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction:")
    st.success(class_names[class_index])

    st.subheader("Confidence:")
    st.write(f"{confidence * 100:.2f}%")