import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import os
import tempfile
import requests
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("🧥 Fashion Recommender System 👗")

# ------------------------- Download from Google Drive -------------------------
@st.cache_resource
def download_file_from_gdrive(gdrive_id, dest_path):
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    r = requests.get(url, allow_redirects=True)
    with open(dest_path, 'wb') as f:
        f.write(r.content)
    return dest_path

# Google Drive file IDs (you must set your own here)
EMBEDDINGS_ID = '1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt'
FILENAMES_ID = '1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW'

# Temporary directory to store files
temp_dir = tempfile.gettempdir()
embeddings_path = os.path.join(temp_dir, "embeddings.pkl")
filenames_path = os.path.join(temp_dir, "filenames.pkl")

# Download files if not already cached
download_file_from_gdrive(EMBEDDINGS_ID, embeddings_path)
download_file_from_gdrive(FILENAMES_ID, filenames_path)

# ------------------------- Load Data -------------------------
with open(embeddings_path, "rb") as f:
    feature_list = pickle.load(f)

with open(filenames_path, "rb") as f:
    filenames = pickle.load(f)

# Convert to numpy array
feature_list = np.array(feature_list)

# ------------------------- Helper Function -------------------------
def save_uploaded_image(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            return temp_file.name
    except Exception as e:
        st.error(f"Failed to save uploaded image: {e}")
        return None

def recommend(image_path, feature_list, filenames, top_k=5):
    model = NearestNeighbors(n_neighbors=top_k, algorithm='brute', metric='euclidean')
    model.fit(feature_list)

    img = Image.open(image_path).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, -1)

    distances, indices = model.kneighbors(img_array)
    return indices

# ------------------------- UI -------------------------
st.markdown("### Upload an image of a clothing item to get similar suggestions")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
top_k = st.slider("Number of Recommendations", 1, 10, 5)

if uploaded_file:
    # Show preview
    image_path = save_uploaded_image(uploaded_file)
    if image_path:
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        # Show Recommendations
        indices = recommend(image_path, feature_list, filenames, top_k=top_k)

        st.markdown("### 🔍 Recommended Items")
        cols = st.columns(top_k)
        for i, col in zip(indices[0], cols):
            with col:
                try:
                    rec_image = Image.open(filenames[i])
                    st.image(rec_image, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image {filenames[i]}: {e}")

# embeddings ---  https://drive.google.com/file/d/1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt/view?usp=sharing
# filename ---- https://drive.google.com/file/d/1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW/view?usp=sharing




