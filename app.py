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
            background: rgba(255,255,255,0.95);
            color: #111827;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid #dbe4dc;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
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
    st.markdown("## ⚙️ Controls")
    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )
    st.markdown("### Model info")
    st.write("• Input size: 224 × 224")
    st.write(f"• Classes: {len(class_names)}")
    st.write("• Output: disease label + confidence")

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