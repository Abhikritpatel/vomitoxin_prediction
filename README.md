
# Vomitoxin Prediction Using Hyperspectral imaging data

This repository contains a machine learning model for predicting vomitoxin levels using deep learning.

# Project Overview

Vomitoxin (Deoxynivalenol) is a mycotoxin that affects grain crops. This project leverages a trained deep learning model (best_vomitoxin_model.h5) to predict vomitoxin levels from Hyperspectral imaging data

# Features

1.Deep learning-based vomitoxin prediction.  
2.Model trained using TensorFlow/Keras.  
3.Dataset preprocessing and feature extraction included.  

# Repository Structure
vomitoxin_prediction:   
│-- best_vomitoxin_model.h5 (Trained deep learning model).   
│-- requirements.txt (Dependencies).    
│-- README.md  (Project documentation).   
│-- data (Folder for input datasets).    
│-- notebooks (Jupyter notebooks for data analysis and the model).   

# Model Detail
Architecture: Deep Learning (CNN/LSTM/etc.).  
Framework: TensorFlow/Keras.  
Training Data: Hyperspectral imaging data for corn samples at different wavelengths and their corresponding vomitoxin concentration

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Abhikritpatel/vomitoxin_prediction.git  
cd vomitoxin_prediction  
pip install -r requirements.txt  

```
    
You can load the trained model and use it to make predictions with new input data:

```bash
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("best_vomitoxin_model.h5")

# Example: Predict using sample input (replace with actual data)
import numpy as np
sample_input = np.array([[feature1, feature2, feature3, ...]])  # Replace with actual input features
prediction = model.predict(sample_input)
print("Predicted Vomitoxin Level:", prediction)

```

