import streamlit as st
import torch
import gdown
import os
from PIL import Image
import cv2
import numpy as np

# Page config
st.set_page_config(page_title="Wildlife Detector - YOLOv5m6u", layout="centered")

# Load model
@st.cache_resource
def load_model():
    file_id = "16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ"  # 🔁 Replace with your actual file ID
    url = f"https://drive.google.com/uc?id=16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ"
    model_path = "saif.pt"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=False)
    return model

model = load_model()

# UI
st.title("🐾 Wildlife Detector with YOLOv5m6u")
st.write("Upload an image and detect wildlife using your YOLOv5m6u model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Detecting..."):
        results = model(image_bgr)
        results.render()
        detected_img = results.imgs[0]
        st.image(detected_img[:, :, ::-1], caption="Detection Result", use_container_width=True)

    st.success("Detection complete!")
