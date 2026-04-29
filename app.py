# import streamlit as st

# st.title("🌿 Plant Disease Detection")
# st.write("Upload a leaf image to detect disease")

# # Load your model
# from tensorflow.keras.models import load_model

# model = load_model("plant_disease_final (1).keras")

# # Image upload
# from PIL import Image

# uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_container_width=True)

# # Preprocess image
# import numpy as np

# def preprocess_image(image):
#     image = image.resize((224, 224))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# class_names = [
#  'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
#  'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy',
#  'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
#  'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)', 'Grape___healthy',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot', 'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
#  'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
#  'Raspberry___healthy', 'Soybean___healthy',
#  'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
#  'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
#  'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
# ]

# # Prediction logic
# if uploaded_file is not None:
#     processed_img = preprocess_image(image)

#     prediction = model.predict(processed_img)
#     class_index = np.argmax(prediction)
#     confidence = np.max(prediction)

#     st.subheader("Prediction:")
#     st.success(class_names[class_index])

#     st.subheader("Confidence:")
#     st.write(f"{confidence * 100:.2f}%")

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import plotly.express as px
import time

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme / CSS ----------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f4f7f5 0%, #e9f1ea 100%);
            color: #1f2937;
        }

        /* Main header card */
        .hero {
            padding: 1.5rem 1.8rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #123524 0%, #1f5c3b 100%);
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            color: white;
        }
        .hero p {
            margin: 0.35rem 0 0 0;
            color: #e8f3ea;
            font-size: 1rem;
        }

        /* Cards */
        .metric-card, .section-card {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0;
        }

        /* Text helpers */
        .small-muted {
            color: #4b5563;
            font-size: 0.92rem;
        }

        /* Streamlit text */
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: inherit;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        /* Buttons */
        .stButton button {
            background: #166534;
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }
        .stButton button:hover {
            background: #14532d;
            color: white;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            # background: #e5efe7;
            color: #123524 !important;
            border-radius: 10px 10px 0 0;
            padding: 10px 16px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: #000000 !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: #cde8d4;
            color: #123524 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Model ----------
@st.cache_resource
def load_plant_model():
    return load_model("plant_disease_final (1).keras")

model = load_plant_model()

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


disease_info = {

# ---------------- APPLE ----------------
"Apple___Apple_scab": {
    "description": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
    "treatment": "Remove fallen leaves, prune infected branches, and apply fungicide sprays during early growth stages. Improve air circulation around trees."
},

"Apple___Black_rot": {
    "description": "Fungal infection causing dark rot on fruits and leaf spots.",
    "treatment": "Prune dead wood, remove infected fruits, and use protective fungicides. Avoid water stress in trees."
},

"Apple___Cedar_apple_rust": {
    "description": "Fungal disease causing orange-yellow spots on leaves.",
    "treatment": "Remove nearby cedar trees if possible, apply fungicide during early spring, and prune infected leaves."
},

"Apple___healthy": {
    "description": "The apple plant shows no signs of disease.",
    "treatment": "Maintain proper watering, pruning, and balanced fertilization."
},

# ---------------- BLUEBERRY ----------------
"Blueberry___healthy": {
    "description": "Healthy blueberry plant without visible disease.",
    "treatment": "Ensure acidic soil, proper irrigation, and regular monitoring."
},

# ---------------- CHERRY ----------------
"Cherry_(including_sour)___Powdery_mildew": {
    "description": "White powder-like fungal growth on leaves and shoots.",
    "treatment": "Apply sulfur-based fungicides, improve air circulation, and remove infected shoots."
},

"Cherry_(including_sour)___healthy": {
    "description": "Healthy cherry plant.",
    "treatment": "Maintain proper pruning and nutrient supply."
},

# ---------------- CORN ----------------
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
    "description": "Fungal disease causing gray lesions on corn leaves.",
    "treatment": "Use resistant varieties, rotate crops, and apply fungicides during early infection."
},

"Corn_(maize)___Common_rust_": {
    "description": "Reddish-brown pustules appear on leaves.",
    "treatment": "Plant resistant hybrids and apply fungicide if severe infection occurs."
},

"Corn_(maize)___Northern_Leaf_Blight": {
    "description": "Long gray-green lesions that spread rapidly on leaves.",
    "treatment": "Crop rotation, resistant seeds, and timely fungicide application."
},

"Corn_(maize)___healthy": {
    "description": "Healthy corn plant.",
    "treatment": "Maintain proper irrigation and soil nutrition."
},

# ---------------- GRAPE ----------------
"Grape___Black_rot": {
    "description": "Fungal disease causing black lesions on fruit and leaves.",
    "treatment": "Remove infected fruits, prune vines, and apply fungicide sprays during growing season."
},

"Grape___Esca_(Black_Measles)": {
    "description": "Wood disease causing leaf discoloration and fruit rot.",
    "treatment": "Remove infected vines, avoid pruning wounds, and apply protective treatments."
},

"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
    "description": "Brown spots leading to leaf drying.",
    "treatment": "Use fungicides and ensure proper vineyard ventilation."
},

"Grape___healthy": {
    "description": "Healthy grape plant.",
    "treatment": "Maintain proper pruning, irrigation, and pest monitoring."
},

# ---------------- ORANGE ----------------
"Orange___Haunglongbing_(Citrus_greening)": {
    "description": "Severe bacterial disease causing yellow shoots and bitter fruits.",
    "treatment": "Remove infected trees, control insect vectors, and use certified healthy plants."
},

# ---------------- PEACH ----------------
"Peach___Bacterial_spot": {
    "description": "Bacterial infection causing dark spots on leaves and fruit.",
    "treatment": "Use copper-based sprays, avoid overhead irrigation, and prune infected branches."
},

"Peach___healthy": {
    "description": "Healthy peach plant.",
    "treatment": "Maintain balanced fertilization and pest control."
},

# ---------------- PEPPER ----------------
"Pepper,_bell___Bacterial_spot": {
    "description": "Bacterial disease causing dark leaf spots and fruit lesions.",
    "treatment": "Use disease-free seeds, copper sprays, and avoid wet foliage."
},

"Pepper,_bell___healthy": {
    "description": "Healthy pepper plant.",
    "treatment": "Ensure proper sunlight and watering schedule."
},

# ---------------- POTATO ----------------
"Potato___Early_blight": {
    "description": "Brown concentric spots on leaves caused by fungus.",
    "treatment": "Apply fungicides, rotate crops, and remove infected leaves."
},

"Potato___Late_blight": {
    "description": "Rapidly spreading disease causing leaf and tuber decay.",
    "treatment": "Apply fungicides immediately, remove infected plants, and avoid overhead watering."
},

"Potato___healthy": {
    "description": "Healthy potato plant.",
    "treatment": "Maintain soil fertility and proper irrigation."
},

# ---------------- SOYBEAN ----------------
"Soybean___healthy": {
    "description": "Healthy soybean plant.",
    "treatment": "Maintain crop rotation and proper nutrient management."
},

# ---------------- SQUASH ----------------
"Squash___Powdery_mildew": {
    "description": "White powder-like fungus on leaves.",
    "treatment": "Use fungicides, improve air circulation, and avoid overcrowding plants."
},

# ---------------- STRAWBERRY ----------------
"Strawberry___Leaf_scorch": {
    "description": "Fungal disease causing brown leaf edges and spotting.",
    "treatment": "Remove infected leaves, apply fungicides, and improve airflow between plants."
},

"Strawberry___healthy": {
    "description": "Healthy strawberry plant.",
    "treatment": "Maintain proper watering and nutrient balance."
},

# ---------------- TOMATO ----------------
"Tomato___Bacterial_spot": {
    "description": "Bacterial disease causing dark spots on leaves and fruit.",
    "treatment": "Use copper sprays, avoid overhead irrigation, and remove infected plants."
},

"Tomato___Early_blight": {
    "description": "Fungal disease causing concentric ring spots on leaves.",
    "treatment": "Apply fungicides, rotate crops, and remove infected foliage."
},

"Tomato___Late_blight": {
    "description": "Severe fungal disease causing dark lesions and plant collapse.",
    "treatment": "Apply fungicides immediately, remove infected plants, and avoid wet conditions."
},

"Tomato___Leaf_Mold": {
    "description": "Fungal disease causing yellow spots and mold under leaves.",
    "treatment": "Improve ventilation, reduce humidity, and use fungicides."
},

"Tomato___Septoria_leaf_spot": {
    "description": "Small dark spots that spread rapidly on leaves.",
    "treatment": "Remove infected leaves, use fungicides, and avoid overhead watering."
},

"Tomato___Spider_mites Two-spotted_spider_mite": {
    "description": "Pest infestation causing yellow speckling on leaves.",
    "treatment": "Use miticides or neem oil and maintain humidity control."
},

"Tomato___Target_Spot": {
    "description": "Fungal disease causing circular target-like lesions.",
    "treatment": "Apply fungicides and remove infected plant debris."
},

"Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
    "description": "Virus causing curled yellow leaves and stunted growth.",
    "treatment": "Remove infected plants and control whitefly vectors."
},

"Tomato___Tomato_mosaic_virus": {
    "description": "Virus causing mosaic patterns on leaves.",
    "treatment": "Remove infected plants and disinfect tools."
},

"Tomato___healthy": {
    "description": "Healthy tomato plant.",
    "treatment": "Maintain proper watering, fertilization, and pest monitoring."
}

}

def get_recommendation(label):
    if label in disease_info:
        return disease_info[label]
    else:
        return {
            "description": "No specific information available.",
            "treatment": "Consult an agricultural expert."
        }

# ---------- Helpers ----------
def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def format_label(label):
    return label.replace("___", " • ").replace("_", " ")

def get_disease_type(label):
    if "healthy" in label.lower():
        return "Healthy"
    return "Diseased"

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🌿 Plant Disease Detection</h1>
        <p>Upload a leaf image, analyze it instantly, and view a clean diagnosis dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("# ⚙️ Settings")
    st.markdown("## Model Info")
    st.write("• Input Size: 224 × 224")
    st.write(f"• Disease Types: {len(class_names)}")
    st.write("• Output: Disease Label + Confidence ")
    st.markdown("---")

    st.markdown("# 📸 Support")
    st.markdown("## Image Capture Tips")
    st.write("☀️ Natural Daylight")
    st.write("🚫 Avoid Shadows")
    st.write("🔍 Capture Effected Area Clearly")
    st.write("🍃 Single Leaf Centered in the Image")
    st.write("🌱 Plain Background (sky, soil, hand) ")


# ---------- Centered Uploader ----------
st.markdown("---")

_, c2, _ = st.columns([0.2, 0.6, 0.2])

with c2:
    # 1. Container for the card
    with st.container():
        st.markdown(
            """
            <div style="
                background: #ffffff;
                padding: 30px;
                border-radius: 20px;
                border: 2px dashed #166534;
                text-align: center;
                box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            ">
                <h3 style="color:#123524; margin-top:0;">📤 Upload Leaf Image</h3>
                <p style="color:#4b5563; font-size:0.9rem; margin-bottom: 20px;">
                    Select or drag-and-drop a clear image of the plant leaf.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 2. Use a CSS override to "push" the uploader up into the div
        st.markdown(
            """
            <style>
                /* Target the file uploader within this specific column */
                [data-testid="stFileUploader"] {
                    margin-top: -110px;
                    opacity: 0;
                    cursor: pointer;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.markdown("---")

# ---------- Main Layout ----------
col1, col2 = st.columns([1.05, 0.95], gap="large")

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Image Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.info("Upload a leaf image from the sidebar to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Prediction Panel")

    if uploaded_file is not None:
        try:
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img, verbose=0)[0]
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            predicted_label = class_names[class_index]
            info = get_recommendation(predicted_label)
            disease_type = get_disease_type(predicted_label)

            # REPLACE YOUR EXISTING ST.METRIC OR CUSTOM BOXES WITH THIS:
            st.markdown(
                f"""
                <div style="display:flex; gap: 10px; margin-bottom:15px;">
                    <div style="flex:1; background:#0f172a; color:#ffffff; padding:15px; border-radius:16px;">
                        <div style="font-size:0.8rem; color:#94a3b8;">Prediction</div>
                        <div style="font-size:1.1rem; font-weight:700;">{format_label(predicted_label)}</div>
                    </div>
                    <div style="flex:1; background:#14532d; color:#ffffff; padding:15px; border-radius:16px;">
                        <div style="font-size:0.8rem; color:#bbf7d0;">Confidence</div>
                        <div style="font-size:1.1rem; font-weight:700;">{confidence * 100:.2f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            tab1, tab2 = st.tabs(["Result", "All Probabilities"])

            with tab1:
                # Use custom div instead of st.success/st.error
                color_bg = "#dcfce7" if disease_type == "Healthy" else "#fee2e2"
                color_text = "#14532d" if disease_type == "Healthy" else "#991b1b"
                
                st.markdown(
                    f"""
                    <div style="padding:15px; border-radius:12px; background:{color_bg}; color:{color_text}; font-weight:600; margin-bottom:10px;">
                        {"Healthy leaf detected" if disease_type == "Healthy" else "Disease detected"}: {format_label(predicted_label)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Replace st.write with styled custom rows
                st.markdown(f"**Diagnosis type:** {disease_type}")
                st.markdown(f"**Class index:** {class_index}")
                st.markdown("### Disease Information")
                # --- Styled Disease Info ---
                st.markdown("**Description:**")
                st.markdown(
                    f"""
                    <div style="
                        padding:15px; 
                        border-radius:12px; 
                        background:#f3f4f6; 
                        color:#374151; 
                        border-left:5px solid #6b7280;
                        margin-bottom:15px;
                    ">
                        {info['description']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # --- Styled Treatment (Dark Green) ---
                st.markdown("**Recommended Treatment:**")
                st.markdown(
                    f"""
                    <div style="
                        padding:15px; 
                        border-radius:12px; 
                        background:#e5efe7; 
                        color:#166534; 
                        font-weight:600;
                        border-left:5px solid #166534;
                    ">
                        {info['treatment']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            with tab2:
                top_k = 5
                top_indices = np.argsort(prediction)[::-1][:top_k]
                for idx in top_indices:
                    st.markdown(f"• **{format_label(class_names[idx])}:** {prediction[idx] * 100:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.write("No prediction yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("Built for fast, clean plant disease screening.")