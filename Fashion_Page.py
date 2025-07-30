import streamlit as st
import os
import gdown
from PIL import Image, UnidentifiedImageError
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# --- PATHS & CONFIG ---
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'images'  # Ensure this exists and has recommendation images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMBEDDINGS_ID = "1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt"
FILENAMES_ID = "1KFBKrcEMrojJW3NMrS-58qXumfZcl6Zh"

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🧥 Fashion Recommender System 👗</h1>", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    return model

@st.cache_data
def load_features():
    download_from_drive(EMBEDDINGS_ID, "embeddings.pkl")
    download_from_drive(FILENAMES_ID, "filenames_clean.pkl")
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        filenames = pickle.load(open('filenames_clean.pkl', 'rb'))
        return feature_list, filenames
    except Exception as e:
        st.error(f"Failed to load embeddings or filenames: {e}")
        return None, None

def feature_extraction(img_path, model):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        result = model.predict(img_array, verbose=0).flatten()
        return result / norm(result)
    except UnidentifiedImageError:
        st.error("Uploaded image is not valid or corrupted.")
        return None

def recommend(features, feature_list, k=5):
    neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- MAIN APP LOGIC ---

model = load_model()
feature_list, filenames = load_features()

if 'recent' not in st.session_state:
    st.session_state['recent'] = []

uploaded_file = st.file_uploader("Upload a fashion image (jpg/png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    try:
        st.image(Image.open(file_path), caption="Uploaded Image", use_column_width=True)
    except UnidentifiedImageError:
        st.error("Uploaded image could not be displayed.")

    if file_path not in st.session_state['recent']:
        st.session_state['recent'].append(file_path)

    top_k = st.slider("How many recommendations to show?", 1, 10, 5)

    with st.spinner("Extracting features and finding recommendations..."):
        features = feature_extraction(file_path, model)

        if features is not None:
            if features.shape[0] != feature_list[0].shape[0]:
                st.error("❌ Model mismatch! Please regenerate embeddings using this model.")
            else:
                indices = recommend(features, feature_list, top_k)
                st.subheader("🎯 Recommended Items:")
                cols = st.columns(top_k)
                for i in range(top_k):
                    rec_img_path = filenames[indices[0][i]]
                    rec_img_path = os.path.basename(rec_img_path)
                    full_path = os.path.join(IMAGE_FOLDER, rec_img_path)

                    try:
                        with cols[i]:
                            st.image(full_path, caption=f"Recommendation {i+1}", use_column_width=True)
                            with open(full_path, "rb") as f:
                                st.download_button(f"Download {i+1}", f.read(), file_name=rec_img_path)
                    except Exception:
                        st.warning(f"⚠️ Could not load or download image: {full_path}")

# Recently Viewed Section
if st.session_state['recent']:
    st.subheader("🕒 Recently Viewed Images:")
    try:
        st.image(st.session_state['recent'], width=100)
    except:
        st.warning("Could not load some recently viewed images.")

# embeddings --- 1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt
#clean filenames --- 1KFBKrcEMrojJW3NMrS-58qXumfZcl6Zh
