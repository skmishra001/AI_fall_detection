# AI_fall_detection

# 🌿 AI-Based Plant Disease Detection Using Leaf Images

This project uses a Convolutional Neural Network (CNN) to detect plant diseases from leaf images. Built using Python, TensorFlow, and Flask, this model helps farmers and agriculturists to identify diseases early and take corrective actions.

---

## 📌 Features

- Detects multiple plant diseases using deep learning
- Simple web interface to upload leaf images
- Built using TensorFlow and Flask
- Trained on the PlantVillage dataset

---

## 📂 Dataset

We used the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) available on Kaggle.  
After downloading, extract the contents and rename the folder to `dataset/`.

---

## 📦 Installation

```bash
pip install tensorflow keras opencv-python flask


AI_Plant_Disease_Detection/
│
├── dataset/                     # PlantVillage images (downloaded manually)
├── train_model.py               # Model training script
├── predict.py                   # Image prediction script
├── plant_disease_model.h5       # Trained model (after training)
├── app.py                       # Flask web server
├── templates/
│   └── index.html               # Web interface
└── README.md                    # Project documentation
