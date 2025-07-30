import os
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from numpy.linalg import norm

# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract feature vector
def extract_features(img_path, model):
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')  # Ensure correct format
        img = img.resize((224, 224))  # Resize for MobileNetV2
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img, verbose=0).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except UnidentifiedImageError:
        print(f"❌ Unidentified image: {img_path}")
    except Exception as e:
        print(f"⚠️ Skipping {img_path} due to error: {e}")
    return None

# Path to image folder
images_dir = r"images"

# Collect valid image filenames
all_files = os.listdir(images_dir)
image_extensions = ('.jpg', '.jpeg', '.png')
filenames = [os.path.join(images_dir, f) for f in all_files if f.lower().endswith(image_extensions)]

# Extract features
feature_list = []
valid_filenames = []

for file in tqdm(filenames, desc="🔍 Extracting features"):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)
        valid_filenames.append(file)

# Save only successfully processed entries
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(valid_filenames, open('filenames.pkl', 'wb'))

print(f"\n✅ Saved {len(feature_list)} feature vectors and filenames.")




print("Feature shape:", features.shape)
print("First embedding shape:", feature_list[0].shape)
