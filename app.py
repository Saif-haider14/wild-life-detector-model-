import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import gdown

# ========== CONFIG ========== #
MODEL_FILE = "best1.pt"
FILE_ID = "16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ"

# ✅ Set page config
st.set_page_config(page_title="Wildlife Detection", layout="centered")

# ✅ Set background image
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

background_url = "https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9yZXN0JTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%3D%3D"
set_background(background_url)

# ✅ Title and Instructions
st.title("🦓🐃 YOLOv5m6u Detection App 🦏🐘")
st.markdown("Upload an image to detect wildlife using your trained YOLOv5m6u model.")

# ✅ Download model if not available
if not os.path.exists(MODEL_FILE):
    with st.spinner("🔄 Downloading model from Google Drive..."):
        try:
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_FILE, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error("❌ Model download failed. Please check the Google Drive link or access permissions.")
            st.stop()

# ✅ Load model with caching
@st.cache_resource
def load_model():
    return YOLO(MODEL_FILE)

model = load_model()

# ✅ Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])

# ✅ Run detection
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Running Detection..."):
        results = model(image)
        results[0].save(filename="result.jpg")

    st.success("✅ Detection Complete")
    st.image("result.jpg", caption="🎯 Detection Result", use_container_width=True)
