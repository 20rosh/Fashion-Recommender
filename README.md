Fashion Recommender System
A content-based fashion recommendation system built with Streamlit, TensorFlow MobileNetV2, and k-NN.
Upload a fashion image and get visually similar recommendations based on deep learning image features.

🚀 Live Demo
Try the App on Streamlit Cloud:(https://fashion-recommender-xvvrp3jrpcd4y4aukjp4gk.streamlit.app/)

🎯 Features
- Upload fashion images (.jpg/.png)
- Feature extraction using pretrained MobileNetV2
- Similar item search using k-Nearest Neighbors
- Adjustable Top-K slider for number of recommendations
- Download recommended images

🛠️ Tech Stack
- Frontend: Streamlit
- Feature Extraction: MobileNetV2 (TensorFlow / Keras)
- Similarity Search: Scikit-learn (k-NN)
- Image Handling: Pillow, NumPy

📁 Project Structure
fashion-recommender/
├── Fashion_Page.py          # Main Streamlit app
├── app.py                   # Feature extraction script
├── images/                  # Folder of fashion images
├── uploads/                 # Temporary folder for uploaded images
├── embeddings.pkl           # Saved feature vectors
├── filenames.pkl            # Corresponding image paths
├── requirements.txt         # Required Python packages
└── README.md                # Project overview and instructions

▶️ How to Run Locally
1. Clone the repository:
git clone https://github.com/your-username/fashion-recommender.git
cd fashion-recommender

2. Install dependencies:
pip install -r requirements.txt

3. Extract features from images (run once):
python app.py

4. Start the app:
streamlit run Fashion_Page.py

Visit: http://localhost:8501

📸 Screenshots
(Add screenshots of the upload and recommendation sections here)
👩‍💻 Author
Roshani Singh
Email: roshanisingh2005@gmail.com
GitHub:  (https://github.com/20rosh)

