import streamlit as st
import numpy as np
import os
import pickle
import gdown
from PIL import Image
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D

# --- Page config ---
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("👗 Fashion Recommender System")

# --- Google Drive file download ---
EMBEDDINGS_ID = "1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt"
FILENAMES_ID = "1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW"

if not os.path.exists("embeddings.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt}", "embeddings.pkl", quiet=False)
if not os.path.exists("filenames.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW}", "filenames.pkl", quiet=False)

# --- Load features and filenames ---
with open("embeddings.pkl", "rb") as f:
    feature_list = pickle.load(f)

with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# --- Load model ---
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# --- Feature extraction ---
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    return result / np.linalg.norm(result)

# --- Recommendation ---
def recommend(img_path, feature_list, model):
    features = extract_features(img_path, model)
    similarities = cosine_similarity([features], feature_list)[0]
    indices = np.argsort(similarities)[-5:][::-1]
    return [filenames[i] for i in indices]

# --- Upload section ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(temp_path, caption="📸 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Finding recommendations..."):
        results = recommend(temp_path, feature_list, model)

    st.subheader("🛍️ You may also like:")
    cols = st.columns(5)
    for i, file_path in enumerate(results):
        with cols[i]:
            st.image(file_path, use_container_width=True)


# embeddings ---  1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt
# filename ---- 1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW

# git add Fashion_Page.py requirements.txt runtime.txt .gitignore
 #git commit -m "Update Fashion_Page.py with Google Drive integration and recommender system"
 #git push origin main

#   streamlit run 'E:\Teach maven AI-ML\projects\FINAL_FASHION\Fashion_Page.py'