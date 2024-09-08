# Identifying Recyclable Items from Waste using Deep Learning

This project aims to classify waste images as **recyclable** or **non-recyclable** using deep learning techniques, specifically leveraging the **ResNet50** model with **transfer learning**.

## Technologies Used
- **Python**
- **TensorFlow & Keras**
- **ResNet50** for image classification
- **Flask** for web application deployment

## Model Overview
- **ResNet50** pretrained on **ImageNet** is fine-tuned for the waste classification task.
- The model is trained on a dataset of waste images using **image augmentation** for better generalization.
- Achieved an accuracy of **87%** on the test set.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Vaishnavi53/Identifying-Recyclable-Items-From-Images
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the app :
   ```bash
   python app.py
Now upload images through the web interface to classify waste.
   
