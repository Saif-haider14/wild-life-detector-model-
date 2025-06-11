
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile 
import os
import gdown

# ========== CONFIG ========== #
MODEL_FILE = "best1.pt"
FILE_ID = "16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ"




# ✅ Set page config first
st.set_page_config(page_title="Wildlife Detection", layout="centered")

# ✅ Background image from URL
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Add your image URL here
background_url = "https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9yZXN0JTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%3D%3D"
set_background(background_url)

# ✅ Title and UI
st.title("🦓🐃 YOLOv5m6u Detection App"🦏🐘)
st.markdown("Upload an image to detect objects using your trained YOLOv5m6u model.")

# ========== DOWNLOAD MODEL IF NEEDED ========== #
if not os.path.exists(MODEL_FILE):
    with st.spinner("Downloading model from Google Drive..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)
    st.success("Model downloaded successfully!")


# ✅ Load YOLOv5m6u model
@st.cache_resource
def load_model():
    return YOLO("best1.pt")  

model = load_model()

# ✅ Upload and detect
uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Running Detection..."):
        results = model(image)
        results[0].save(filename="result.jpg")

    st.success("✅ Detection Complete")
    st.image("result.jpg", caption="🎯 Detection Result", use_container_width=True)
