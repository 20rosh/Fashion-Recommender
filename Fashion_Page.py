import streamlit as st
import numpy as np
import os
import pickle
from PIL import Image
import gdown
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import load_model

st.set_page_config(page_title="👗 Fashion Recommender", layout="centered")

st.title("👗 Fashion Recommender System")
st.write("Upload a clothing image to get similar fashion recommendations.")

# ---------- STEP 1: Download features if not available ----------
features_file = "features.pkl"
filenames_file = "filenames.pkl"

if not os.path.exists(features_file):
    st.info("🔄 Downloading required feature files...")
    gdown.download("https://drive.google.com/uc?1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt", features_file, quiet=False)

if not os.path.exists(filenames_file):
    gdown.download("https://drive.google.com/uc?id=1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW", filenames_file, quiet=False)

# ---------- STEP 2: Load features ----------
with open(features_file, "rb") as f:
    feature_list = pickle.load(f)

with open(filenames_file, "rb") as f:
    filenames = pickle.load(f)

# ---------- STEP 3: Load the model ----------
model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ---------- STEP 4: Recommendation function ----------
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    return result / np.linalg.norm(result)

def recommend(img_path, feature_list, model):
    features = extract_features(img_path, model)
    similarities = cosine_similarity([features], feature_list)[0]
    indices = np.argsort(similarities)[-5:][::-1]
    return [filenames[i] for i in indices]

# ---------- STEP 5: Upload and Display ----------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save image temporarily
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.image(temp_path, caption="📸 Uploaded Image", use_container_width=True)

    # Find and show recommendations
    with st.spinner("🔍 Finding recommendations..."):
        results = recommend(temp_path, feature_list, model)

    st.success("🎯 Recommendations Found!")
    st.subheader("🛍️ You may also like:")
    cols = st.columns(5)

    for i, path in enumerate(results):
        with cols[i]:
            st.image(path, use_container_width=True)


# embeddings ---  1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt
# filename ---- 1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW

# git add Fashion_Page.py requirements.txt runtime.txt .gitignore
 #git commit -m "Update Fashion_Page.py with Google Drive integration and recommender system"
 #git push origin main

#   streamlit run 'E:\Teach maven AI-ML\projects\FINAL_FASHION\Fashion_Page.py'