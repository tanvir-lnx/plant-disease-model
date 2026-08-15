import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

@st.cache_resource
def load_models():
    mlp_model = joblib.load('plant_disease_mlp_model.pkl')
    scaler = joblib.load('plant_disease_scaler.pkl')
    cnn_model = load_model('plant_disease_cnn.h5')
    return mlp_model, scaler, cnn_model

mlp_model, scaler, cnn_model = load_models()

CLASS_NAMES = ['Tomato Early Blight', 'Tomato Late Blight', 'Tomato Healthy']

def auto_crop_leaf(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([10, 20, 20])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return img_rgb[y:y+h, x:x+w]

def extract_features(img_rgb):
    img_cropped = auto_crop_leaf(img_rgb)
    mean_r, mean_g, mean_b = np.mean(img_cropped[:,:,0]), np.mean(img_cropped[:,:,1]), np.mean(img_cropped[:,:,2])
    std_r, std_g, std_b = np.std(img_cropped[:,:,0]), np.std(img_cropped[:,:,1]), np.std(img_cropped[:,:,2])
    grayscale = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(grayscale)
    contrast = np.std(grayscale)
    edges = cv2.Canny(grayscale, 100, 200)
    edge_density = np.mean(edges) / 255.0
    return np.array([[mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast, edge_density]])

st.title("🌿 Tomato Leaf Disease Detector")
st.write("Upload a tomato leaf photo to check for Early Blight, Late Blight, or Healthy status.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic MLP Model")
            raw_features = extract_features(img_rgb)
            scaled_features = scaler.transform(raw_features)
            idx = mlp_model.predict(scaled_features)[0]
            confidence = np.max(mlp_model.predict_proba(scaled_features)) * 100
            st.success(CLASS_NAMES[idx])
            st.write(f"Confidence: {confidence:.2f}%")

        with col2:
            st.subheader("Advanced CNN Model")
            img_resized = cv2.resize(img_rgb, (64, 64))
            img_normalized = img_resized / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=0)
            predictions = cnn_model.predict(img_expanded)
            idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            st.success(CLASS_NAMES[idx])
            st.write(f"Confidence: {confidence:.2f}%")