%%writefile README.md
# 🚗 CNN Car Classification System

A Convolutional Neural Network (CNN) based image classification project that identifies cars as:

- Suzuki
- Toyota
- Others

## Project Overview

This project uses a CNN model to classify uploaded car images.

The model accepts images of size 150 × 150 pixels and performs image normalization before prediction.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Albumentations
- Streamlit
- Jupyter Notebook

## CNN Architecture

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Dropout
- Flatten
- Dense
- Dropout
- Softmax Output

## Classes

1. Others
2. Suzuki
3. Toyota

## Streamlit Application

The Streamlit application allows users to:

1. Upload a car image
2. Preprocess the image
3. Run the CNN model
4. Predict the car category
5. Display prediction confidence

## Project Structure

```text
CNN_Streamlit_Project/
│
├── app.py
├── Gopi CNN Project.ipynb
├── model.keras
├── label_encoder.pkl
├── requirements.txt
├── README.md
└── results/
