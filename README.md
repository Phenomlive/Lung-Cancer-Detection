

# Lung Cancer Detection

A machine learning project aimed at detecting lung cancer from CT scan images using a Convolutional Neural Network (CNN). This project leverages deep learning techniques to classify CT scans as cancerous or non-cancerous, contributing to early diagnosis and improved patient outcomes.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About
This repository contains a deep learning-based system for detecting lung cancer from CT scan images. The project utilizes a Convolutional Neural Network (CNN) to analyze chest CT scans and classify them into categories such as Normal or Cancerous. The model is designed to assist in early detection, which is critical for improving survival rates, as lung cancer is a leading cause of cancer-related deaths worldwide. The project is built using Python, TensorFlow, and Keras, and it processes medical imaging data to support clinical decision-making.[](https://github.com/ayush9304/Lung_Cancer_Detection)

## Features
- **CNN-Based Classification**: Employs a Convolutional Neural Network to classify CT scan images as cancerous or non-cancerous.
- **Image Preprocessing**: Includes preprocessing steps to handle CT scan images, such as normalization and resizing.
- **High Accuracy**: Aims to achieve high accuracy in detecting lung cancer, with potential for reducing false positives.
- **Scalable Pipeline**: Designed for integration into larger medical imaging workflows or web applications.
- **User-Friendly Interface**: Supports easy model training and evaluation via Jupyter notebooks.

## Installation
To set up the Lung Cancer Detection project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Phenomlive/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required packages:
   ```bash
   pip install tensorflow keras opencv-python numpy pandas
   pip install -r requirements.txt
   ```

4. **Download Dataset**:
   The project uses CT scan images (e.g., from the LIDC-IDRI or LUNA16 datasets). Download the dataset from a source like [The Cancer Imaging Archive](http://www.cancerimagingarchive.net/) or Kaggle and place it in a `data/` folder within the repository. Update the dataset path in the code as needed.[](https://github.com/VinayBN8997/Lung_Cancer_Detection_Using_Python)

## Usage
- **Train the Model**:
  Use the provided Jupyter notebook (e.g., `lung_cancer_detection.ipynb`) to train the CNN model:
  1. Open the notebook in Jupyter:
     ```bash
     jupyter notebook lung_cancer_detection.ipynb
     ```
  2. Follow the steps to load the dataset, preprocess images, and train the model.
  3. Save the trained model for future use.

- **Make Predictions**:
  Use the trained model to classify new CT scan images:
  1. Update the image path in the prediction script or notebook.
  2. Run the prediction code to classify images as Normal or Cancerous.

- **Evaluate the Model**:
  Check the model’s performance metrics (e.g., accuracy, precision, recall) in the notebook or evaluation script.

## Project Structure
- `lung_cancer_detection.ipynb`: Jupyter notebook containing code for data preprocessing, model training, and evaluation.
- `app.py`: Flask application for deploying the model via a web interface.
- `index.html`: HTML template for the web interface to upload and classify CT scan images.
- `static/`: Directory for static files (CSS, JavaScript, images).
  - `css/style.css`: Custom styles for the web interface.
  - `js/script.js`: JavaScript for frontend interactivity.
- `data/`: Directory for storing CT scan datasets (not included in the repository).
- `models/`: Directory for saving trained CNN models.
- `requirements.txt`: Lists Python dependencies.
- `README.md`: This file, providing project documentation.

## Contributing
Contributions are welcome to improve the model’s accuracy, add features, or enhance the web interface. To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Create a pull request.

Please follow Python and TensorFlow coding guidelines and include tests for new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The Cancer Imaging Archive for providing access to CT scan datasets.[](https://github.com/VinayBN8997/Lung_Cancer_Detection_Using_Python)
- The open-source community for tools like TensorFlow, Keras, and Flask.
- Inspiration from the Kaggle Data Science Bowl 2017 and LUNA16 challenge for lung cancer detection research.[](https://github.com/katyaputintseva/LungCancer)

