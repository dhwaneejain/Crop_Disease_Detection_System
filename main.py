import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import os
import cv2

# ---------------------------------
# 🌿 Set page configuration
# ---------------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# ---------------------------------
# 🎨 CSS Styling
# ---------------------------------
st.markdown("""
    <style>
        body { background-color: #f4fff6; font-family: "Segoe UI", sans-serif; }
        h1, h2, h3 { color: #22793c; }
        .stButton > button {
            background-color: #4CAF50; color: white; border-radius: 8px;
        }
        .stTextInput > div > input {
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# 🧠 Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    model_path = 'trained_model.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please add 'trained_model.h5' to the app directory.")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

# ---------------------------------
# 🧠 Prediction Function
# ---------------------------------
def predict_image(image):
    image = Image.open(image).convert("RGB")
    image = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# ---------------------------------
# 🌿 Class Labels + Actions
# ---------------------------------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Dummy suggestions
suggested_actions = {
    label: ["Use fungicide", "Remove infected leaves"] if "healthy" not in label else ["No action needed", "Keep monitoring"]
    for label in class_names
}

# ---------------------------------
# 🚀 Navigation
# ---------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📖 About", "📁 Upload & Predict", "📷 Camera & Predict"])

# ---------------------------------
# 🏠 Home Page
# ---------------------------------
if page == "🏠 Home":
    st.title("🌿 Welcome to Plant Disease Detector")
    st.write("This app helps you detect plant diseases using AI-powered image classification.")
    st.image(r"C:\Users\Nandini Sharma\Downloads\Minor project\Minor project\dataset\home_page.jpeg", use_container_width=True)
    st.markdown("---")
    st.header("📬 Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        comments = st.text_area("Your Feedback")
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            if name and comments:
                feedback_file = "feedback.xlsx"
                feedback_data = {"Name": [name], "Feedback": [comments], "Time": [datetime.datetime.now()]}
                df = pd.DataFrame(feedback_data)
                if os.path.exists(feedback_file):
                    old_df = pd.read_excel(feedback_file)
                    df = pd.concat([old_df, df], ignore_index=True)
                df.to_excel(feedback_file, index=False)
                st.success("✅ Feedback submitted successfully!")
            else:
                st.warning("Please fill in all fields.")

# ---------------------------------
# 📖 About Page
# ---------------------------------
elif page == "📖 About":
    st.title("📖 About This App")
    st.write("""
        This application is developed as a final year BCA project.  
        It uses a Convolutional Neural Network (CNN) trained on the **PlantVillage dataset**  
        to identify **38 types of plant diseases and healthy leaves**.
        
        Technologies used:
        - Python 🐍
        - TensorFlow / Keras 🧠
        - OpenCV 📷
        - Streamlit 💻
        - Pandas & Excel for feedback 📊
    """)

# ---------------------------------
# 📁 Upload & Predict Page
# ---------------------------------
elif page == "📁 Upload & Predict":
    st.title("📁 Upload Leaf Image")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict"):
            label, confidence = predict_image(uploaded_file)
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence*100:.2f}%")
            st.markdown("### Suggested Actions:")
            for act in suggested_actions[label]:
                st.write(f"- {act}")

# ---------------------------------
# 📷 Camera & Predict Page
# ---------------------------------
elif page == "📷 Camera & Predict":
    st.title("📷 Capture via Camera")
    captured_image = st.camera_input("Take a picture")

    if captured_image is not None:
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        if st.button("Predict"):
            label, confidence = predict_image(captured_image)
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence*100:.2f}%")
            st.markdown("### Suggested Actions:")
            for act in suggested_actions[label]:
                st.write(f"- {act}")

    st.markdown("---")
    st.header("📬 Feedback")
    with st.form("camera_feedback_form"):
        name = st.text_input("Your Name", key="camera_name")
        comments = st.text_area("Your Feedback", key="camera_comments")
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            if name and comments:
                feedback_file = "feedback.xlsx"
                feedback_data = {"Name": [name], "Feedback": [comments], "Time": [datetime.datetime.now()]}
                df = pd.DataFrame(feedback_data)
                if os.path.exists(feedback_file):
                    old_df = pd.read_excel(feedback_file)
                    df = pd.concat([old_df, df], ignore_index=True)
                df.to_excel(feedback_file, index=False)
                st.success("✅ Feedback submitted successfully!")
            else:
                st.warning("Please fill in all fields.")