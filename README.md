# Brain Tumor Detection System

This is a web application that uses machine learning to detect brain tumors in MRI images. The system uses a Random Forest classifier trained on HOG (Histogram of Oriented Gradients) features extracted from brain MRI images to classify whether a tumor is present or not.

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   - Visit: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
   - Download and extract the dataset
   - Create a folder named `brain_tumor_dataset` in the project root
   - Place the training and testing data in `brain_tumor_dataset/training` and `brain_tumor_dataset/testing` respectively

4. Train the model:
   ```bash
   python train_model.py
   ```
   This will create a `brain_tumor_model.joblib` file containing the trained model.

5. Run the Flask application:
   ```bash
   python app.py
   ```

6. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click on the upload area or drag and drop an MRI image
2. The system will process the image and display the results
3. The results will show whether a tumor is detected and the confidence level

## Features

- Modern, responsive web interface
- Drag and drop image upload
- Real-time image processing
- Confidence level visualization
- Support for various image formats (JPG, PNG, JPEG)

## Model Architecture

The system uses:
- HOG (Histogram of Oriented Gradients) for feature extraction
- Random Forest classifier for tumor detection
- Image preprocessing and resizing
- Probability-based confidence scoring

## Requirements

- Python 3.7+
- Flask 2.0.1
- scikit-learn 0.24.2
- OpenCV
- NumPy
- Pillow
- Other dependencies listed in requirements.txt 