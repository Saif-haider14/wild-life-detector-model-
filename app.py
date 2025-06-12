import streamlit as st
import gdown
import tempfile
from PIL import Image
from ultralytics import YOLO

# Google Drive file ID and local model path
FILE_ID = "16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ"
MODEL_PATH = "saif.pt"

@st.cache_resource
def load_model():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    model = YOLO(MODEL_PATH)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("Wildlife Detection with YOLOv5m6u")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    result_image = results[0].plot()  # returns numpy image with boxes drawn
    st.image(result_image, caption="Detection Result", use_container_width=True)
