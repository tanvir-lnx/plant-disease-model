from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 1. LOAD BOTH AI MODELS ---
print("Loading AI Models into memory...")
try:
    # Model A: The original Machine Learning model with tabular features
    mlp_model = joblib.load('plant_disease_mlp_model.pkl')
    scaler = joblib.load('plant_disease_scaler.pkl')
    
    # Model B: The new Deep Learning Convolutional Neural Network
    cnn_model = load_model('plant_disease_cnn.h5')
    
    print("Success! Both models loaded.")
except Exception as e:
    print(f"Error loading model files: {e}")

# The specific classes both models were trained on
CLASS_NAMES = ['Tomato Early Blight', 'Tomato Late Blight', 'Tomato Healthy']

# --- 2. BASIC MODEL PREPROCESSING FUNCTIONS ---
def auto_crop_leaf(img_rgb):
    """
    Uses HSV color thresholding to find the largest green/yellow object 
    and crops the image tightly around it to remove background noise.
    """
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
    """
    Extracts the 9 tabular features required by the basic MLP model.
    """
    # Auto-crop first to isolate the leaf
    img_cropped = auto_crop_leaf(img_rgb)
    
    mean_r = np.mean(img_cropped[:,:,0])
    mean_g = np.mean(img_cropped[:,:,1])
    mean_b = np.mean(img_cropped[:,:,2])
    std_r = np.std(img_cropped[:,:,0])
    std_g = np.std(img_cropped[:,:,1])
    std_b = np.std(img_cropped[:,:,2])
    
    grayscale = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(grayscale)
    contrast = np.std(grayscale)
    edges = cv2.Canny(grayscale, 100, 200)
    edge_density = np.mean(edges) / 255.0 
    
    return np.array([[mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness, contrast, edge_density]])


# --- 3. WEB SERVER ROUTES ---
@app.route('/')
def home():
    # Serves the HTML frontend
    return render_template('index.html')

@app.route('/predict_basic', methods=['POST'])
def predict_basic():
    """
    Endpoint for Model A: The Basic Multi-Layer Perceptron
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract and scale features
        raw_features = extract_features(img_rgb)
        scaled_features = scaler.transform(raw_features)
        
        # Predict
        prediction_index = mlp_model.predict(scaled_features)[0]
        confidence = np.max(mlp_model.predict_proba(scaled_features)) * 100
        
        return jsonify({
            'disease': CLASS_NAMES[prediction_index],
            'confidence': f"{confidence:.2f}%",
            'model_used': 'Basic MLP Model'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    """
    Endpoint for Model B: The Advanced Convolutional Neural Network
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image exactly how the CNN expects it
        img_resized = cv2.resize(img_rgb, (64, 64))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0) # Reshape to (1, 64, 64, 3)
        
        # Predict using Deep Learning
        predictions = cnn_model.predict(img_expanded)
        prediction_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        return jsonify({
            'disease': CLASS_NAMES[prediction_index],
            'confidence': f"{confidence:.2f}%",
            'model_used': 'Advanced CNN Model'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)