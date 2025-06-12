import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import gdown
from PIL import Image

# Google Drive ID for your model (Extract from your shareable link)
FILE_ID = "16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ"  # Replace with actual file ID
MODEL_PATH = "saif.pt"

@st.cache_resource
def download_model():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    return torch.load(MODEL_PATH, map_location=torch.device("cpu"))

# Load model
try:
    model = download_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("Wildlife Detection App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)  # YOLOv5 prediction

    # Plot results
    results.render()
    result_img = Image.fromarray(results.ims[0])
    st.image(result_img, caption="Detected Image", use_column_width=True)
