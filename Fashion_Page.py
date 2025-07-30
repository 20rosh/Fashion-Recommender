import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

st.set_page_config(page_title="Fashion Recommender", layout="wide")  

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Drive file IDs for .pkl files
EMBEDDINGS_ID = "1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt"
FILENAMES_ID = "1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW"

# --- PAGE SETTINGS ---
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🧥 Fashion Recommender System 👗</h1>", unsafe_allow_html=True)

# --- FUNCTIONS ---

def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# App title
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🧥 Fashion Recommender System 👗</h1>", unsafe_allow_html=True)

# Load MobileNetV2
@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    return model

# Load features and filenames
@st.cache_data
def load_features():
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    return feature_list, filenames

model = load_model()
feature_list, filenames = load_features()

# Extract feature vector
def feature_extraction(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    result = model.predict(img_array, verbose=0).flatten()
    return result / norm(result)

# Nearest neighbor search
def recommend(features, feature_list, k=5):
    neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Save uploaded file
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Session state for recently viewed
if 'recent' not in st.session_state:
    st.session_state['recent'] = []

# Upload image
uploaded_file = st.file_uploader("Upload a fashion image (jpg/png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.image(Image.open(file_path), caption="Uploaded Image", use_container_width=True)

    # Add to recently viewed
    if file_path not in st.session_state['recent']:
        st.session_state['recent'].append(file_path)

    # Top-K slider
    top_k = st.slider("How many recommendations to show?", 1, 10, 5)

    with st.spinner("Extracting features and finding recommendations..."):
        features = feature_extraction(file_path, model)
        st.write("Uploaded feature shape:", features.shape)
        st.write("Embedding shape:", feature_list[0].shape)

        if features.shape[0] != feature_list[0].shape[0]:
            st.error("❌ Model mismatch! Your app is using a different model than used to generate embeddings.")
        else:
            indices = recommend(features, feature_list, top_k)
            st.subheader("🎯 Recommended Items:")
            cols = st.columns(top_k)
            for i in range(top_k):
                rec_img_path = filenames[indices[0][i]]
                with cols[i]:
                    st.image(rec_img_path, caption=f"Recommendation {i+1}", use_container_width=True)
                    with open(rec_img_path, "rb") as f:
                        st.download_button(f"Download {i+1}", f.read(), file_name=os.path.basename(rec_img_path))

# Recently viewed
if st.session_state['recent']:
    st.subheader("🕒 Recently Viewed Images:")
    st.image(st.session_state['recent'], width=100)


# embeddings ---  https://drive.google.com/file/d/1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt/view?usp=sharing
# filname ---- https://drive.google.com/file/d/1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW/view?usp=sharing




