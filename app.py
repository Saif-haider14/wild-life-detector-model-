import streamlit as st
import gdown
import os
from ultralytics import YOLO
from PIL import Image
import tempfile

# ------------------ Configuration ------------------
# Replace with your actual Google Drive file ID
file_id = '16p2yZOPplA4BopdyPeJOr2bTWOePc2gQ'  # e.g., '1a2b3c4D5EfGhIJKlMNopQRS6789'
model_filename = 'saif.pt'

# ------------------ Download Model ------------------
@st.cache_resource
def download_model():
    if not os.path.exists(model_filename):
        st.info('Downloading model. Please wait...')
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_filename, quiet=False)
    return YOLO(model_filename)

# ------------------ Load Model ------------------
model = download_model()

# ------------------ Streamlit UI ------------------
st.title("🔍 YOLO Object Detection App")
st.write("Upload an image and detect objects using your custom YOLO model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Detecting objects...")

    # Run detection
    results = model(temp_path)
    result_img = results[0].plot()

    st.image(result_img, caption="Detected Image", use_container_width=True)
