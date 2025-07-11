Fashion Recommender System
A content-based fashion recommendation system built with Streamlit, TensorFlow MobileNetV2, and k-NN.
Upload a fashion image and get visually similar recommendations based on deep learning image features.

🚀 Live Demo
Try the App on Streamlit Cloud: https://your-streamlit-app-link.streamlit.app/
(Replace this with your actual app link after deployment)
🎯 Features
- Upload fashion images (.jpg/.png)
- Feature extraction using pretrained MobileNetV2
- Similar item search using k-Nearest Neighbors
- Adjustable Top-K slider for number of recommendations
- Download recommended images
- Recently viewed history with thumbnails

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

☁️ How to Deploy on Streamlit Cloud
1. Push your project to GitHub
2. Go to https://streamlit.io/cloud
3. Create a new app from your repo and select Fashion_Page.py
4. Click Deploy

📸 Screenshots
(Add screenshots of the upload and recommendation sections here)
👩‍💻 Author
Roshani Singh
Email: roshanisingh2005@example.com
GitHub: https://github.com/your-username

📄 License
This project is licensed under the MIT License.
