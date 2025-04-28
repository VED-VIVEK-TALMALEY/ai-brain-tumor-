import os
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

def extract_features(image_path):
    """Extract features from an image using HOG (Histogram of Oriented Gradients)"""
    try:
        print(f"Processing image: {image_path}")
        # Read and resize image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
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
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def load_dataset(base_dir):
    """Load and process the dataset"""
    X = []
    y = []
    
    for label in ['yes', 'no']:
        label_dir = os.path.join(base_dir, label)
        if not os.path.exists(label_dir):
            print(f"Directory not found: {label_dir}")
            continue
            
        print(f"\nProcessing {label} images from {label_dir}")
        files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(files)} images")
        
        for filename in tqdm(files, desc=f"Processing {label} images"):
            img_path = os.path.join(label_dir, filename)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(1 if label == 'yes' else 0)
    
    return np.array(X), np.array(y)

def check_dataset_structure():
    """Check if the dataset is properly organized"""
    required_dirs = [
        'brain_tumor_dataset/training/yes',
        'brain_tumor_dataset/training/no',
        'brain_tumor_dataset/testing/yes',
        'brain_tumor_dataset/testing/no'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist!")
            print("Please organize the dataset as follows:")
            print("brain_tumor_dataset/")
            print("├── training/")
            print("│   ├── yes/  (tumor images)")
            print("│   └── no/   (non-tumor images)")
            print("└── testing/")
            print("    ├── yes/  (tumor images)")
            print("    └── no/   (non-tumor images)")
            sys.exit(1)
        else:
            print(f"Found directory: {dir_path}")

def main():
    print("Starting brain tumor detection model training...")
    
    # Check dataset structure
    print("\nChecking dataset structure...")
    check_dataset_structure()
    
    # Load training data
    print("\nLoading training data...")
    X_train, y_train = load_dataset('brain_tumor_dataset/training')
    
    if len(X_train) == 0:
        print("Error: No training data found!")
        sys.exit(1)
    
    print(f"\nLoaded {len(X_train)} training samples")
    print(f"Positive samples (tumor): {sum(y_train == 1)}")
    print(f"Negative samples (no tumor): {sum(y_train == 0)}")
    
    # Load testing data
    print("\nLoading testing data...")
    X_test, y_test = load_dataset('brain_tumor_dataset/testing')
    
    if len(X_test) == 0:
        print("Error: No testing data found!")
        sys.exit(1)
    
    print(f"\nLoaded {len(X_test)} testing samples")
    print(f"Positive samples (tumor): {sum(y_test == 1)}")
    print(f"Negative samples (no tumor): {sum(y_test == 0)}")
    
    # Train the model
    print("\nTraining the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save the model
    print("\nSaving the model...")
    model_path = 'brain_tumor_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved successfully as '{model_path}'")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 