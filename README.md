Lung Cancer Detection WebApp
Welcome to the Lung Cancer Detection WebApp repository. This project is a web application designed to detect lung cancer from X-ray images using a pre-trained Keras model.

Table of Contents
Introduction
Dataset
Model Training
Deployment
Usage
Installation
Contributing
License
Introduction
This WebApp allows users to upload X-ray images of lungs to determine if they are affected by cancer. The application uses a pre-trained Keras model, deployed using Streamlit, to perform the classification.

Dataset
The dataset used for training the model consists of X-ray images labeled into two categories:

Cancer: 782 PNG images
No Cancer: 214 PNG images
Model Training
The model was trained using Teachable Machine by Google. The trained Keras model was then converted to TensorFlow Lite format for efficient deployment.

Deployment
The web application is built using Streamlit, a Python library for creating web apps. The following code snippet demonstrates how the TFLite model is loaded and used for predictions within the Streamlit app:

python code:

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="cancer_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up Streamlit app
st.title("Cancer Detection from Tissue Images")
st.write("Upload an image to check whether it is cancer-affected or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    # Ensure the image has 3 channels (convert grayscale images to RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if uploaded_file is not None:
    try:
        # Read the image
        image = Image.open(uploaded_file)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Predict
        prediction = predict(preprocessed_image)

        # Display the image and prediction
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        if prediction[0][0] < 0.5:
            st.write("The image is predicted to be **cancer-affected**.")
        else:
            st.write("The image is predicted to be **not cancer-affected**.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
Usage
To use the WebApp, follow these steps:

Clone the repository.
Install the required dependencies.
Run the Streamlit app.
Installation

Clone the repository:
git clone https://github.com/yourusername/lung-cancer-detection-webapp.git
cd lung-cancer-detection-webapp

Install the required dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
