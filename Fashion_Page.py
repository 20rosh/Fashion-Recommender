import streamlit as st
import numpy as np
import os
import pickle
import gdown
from PIL import Image
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("🧥 Fashion Recommender System 👗")

# Constants
IMAGE_DIR = "images"  # folder where all your images are stored
os.makedirs(IMAGE_DIR, exist_ok=True)

# Google Drive File IDs (replace with your actual file IDs)
file_id_embeddings = '1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt'
file_id_filenames = '1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW'

# Download files from Google Drive if not present
if not os.path.exists('embeddings.pkl'):
    gdown.download(f'https://drive.google.com/uc?id={file_id_embeddings}', 'embeddings.pkl', quiet=False)

if not os.path.exists('filenames.pkl'):
    gdown.download(f'https://drive.google.com/uc?id={file_id_filenames}', 'filenames.pkl', quiet=False)

# Load the embeddings and filenames
with open('embeddings.pkl', 'rb') as f:
    feature_list = pickle.load(f)

with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Convert to full paths (relative to IMAGE_DIR)
full_paths = [os.path.join(IMAGE_DIR, fname) for fname in filenames]

# Upload section
uploads = st.file_uploader("📤 Upload a clothing image", type=['jpg', 'jpeg', 'png'])

# Helper function to find similar images
def recommend(image_path, feature_list, full_paths):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array).flatten()
    similarities = cosine_similarity([features], feature_list)[0]
    indices = np.argsort(similarities)[-5:][::-1]

    return [full_paths[i] for i in indices]

# Main logic
if uploads is not None:
    with open("temp_uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="📸 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Finding recommendations..."):
        recommended_files = recommend("temp_uploaded_image.jpg", feature_list, full_paths)

    st.subheader("🎯 Top 5 Similar Recommendations")
    cols = st.columns(5)
    for i, file in enumerate(recommended_files):
        with cols[i]:
            if os.path.exists(file):
                try:
                    img = Image.open(file)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Couldn't open image {file}: {e}")
            else:
                st.error(f"❌ File not found: {file}")

    st.write("🗂 Recommended file paths:")
    for file in recommended_files:
        st.code(file)


# embeddings ---  https://drive.google.com/file/d/1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt/view?usp=sharing
# filename ---- https://drive.google.com/file/d/1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW/view?usp=sharing

# git add Fashion_Page.py requirements.txt runtime.txt .gitignore 
 #git commit -m "Update Fashion_Page.py with Google Drive integration and recommender system"
 #git push origin main

#   streamlit run 'E:\Teach maven AI-ML\projects\FINAL_FASHION\Fashion_Page.py'