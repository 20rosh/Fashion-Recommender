import streamlit as st
import numpy as np
import os
import pickle
from PIL import Image
import gdown
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalMaxPooling2D

# Load features and filenames
feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Function to extract features
def extract_features(img_path, model):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    result = model.predict(img_array).flatten()
    return result / np.linalg.norm(result)

# Recommend similar images
def recommend(image_path, feature_list, filenames):
    features = extract_features(image_path, model)
    similarities = cosine_similarity([features], feature_list)[0]
    indices = np.argsort(similarities)[-5:][::-1]
    return [filenames[i] for i in indices]

# Streamlit UI
st.title("👗 AI Fashion Recommender")
st.markdown("Upload a fashion item image and get similar recommendations!")

# Ensure uploads folder exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Upload image
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    upload_path = os.path.join("uploads", uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(upload_path, caption="📸 Uploaded Image", use_container_width=True)

    # Recommend similar images
    with st.spinner("🔍 Finding recommendations..."):
        recommended_files = recommend(upload_path, feature_list, filenames)

    st.success("🎉 Recommendations found!")
    for file in recommended_files:
        st.image(file, caption=os.path.basename(file), use_container_width=True)



# embeddings ---  https://drive.google.com/file/d/1uxFuOHmjTx3G1z1CbJD7FzzgmbM6fQxt/view?usp=sharing
# filename ---- https://drive.google.com/file/d/1gytquz6wTp4EP5bQC8vWp9n_XSrzkaGW/view?usp=sharing

# git add Fashion_Page.py requirements.txt runtime.txt .gitignore
 #git commit -m "Update Fashion_Page.py with Google Drive integration and recommender system"
 #git push origin main

#   streamlit run 'E:\Teach maven AI-ML\projects\FINAL_FASHION\Fashion_Page.py'