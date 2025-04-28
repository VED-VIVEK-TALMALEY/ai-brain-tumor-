from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import sys
import joblib
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Configure upload folder and file size limits
UPLOAD_FOLDER = 'static/uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Check if model exists
if not os.path.exists('brain_tumor_model.joblib'):
    print("Error: Model file 'brain_tumor_model.joblib' not found!")
    print("Please run train_model.py first to train and save the model.")
    sys.exit(1)

try:
    # Load the trained model
    model = joblib.load('brain_tumor_model.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

def extract_features(image_path):
    """Extract features from an image using HOG (Histogram of Oriented Gradients)"""
    try:
        # Read and resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        img = cv2.resize(img, (150, 150))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        win_size = (150, 150)
        block_size = (30, 30)
        block_stride = (15, 15)
        cell_size = (15, 15)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(gray)
        
        return features.flatten()
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'})
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            
            try:
                # Extract features from the image
                features = extract_features(file_path)
                if features is None:
                    return jsonify({'error': 'Could not process the image'})
                
                # Make prediction
                prediction = model.predict_proba([features])[0]
                result = 'Tumor Detected' if prediction[1] > 0.5 else 'No Tumor Detected'
                confidence = float(prediction[1]) if prediction[1] > 0.5 else float(1 - prediction[1])
                
                return jsonify({
                    'result': result,
                    'confidence': round(confidence * 100, 2)
                })
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'})
            finally:
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True) 