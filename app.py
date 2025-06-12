import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import gdown
from PIL import Image
from torch.serialization import add_safe_globals
import ultralytics.nn.tasks  # Ensure DetectionModel is imported

# Allow YOLOv5 DetectionModel class to be loaded from .pt
add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# Google Drive file ID (from your share link)
FILE_ID = "
16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ"
MODEL_PATH = "saif.pt"

@st.cache_resource
def download_model():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
    return model

try:
    model = download_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("Wildlife Detection with YOLOv5m6u")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)  # Inference

    # Draw boxes on the image (YOLOv5 renders in-place)
    results.render()
    result_img = Image.fromarray(results.ims[0])
    st.image(result_img, caption="Detection Result", use_column_width=True)
