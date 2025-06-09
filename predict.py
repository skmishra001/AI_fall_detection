from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("plant_disease_model.h5")

class_names = ["Apple___Scab", "Apple___Black_rot", "Apple___healthy", "Corn___Common_rust", "Tomato___Early_blight"]

def predict_leaf(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    return class_names[index]

# Example usage
# print(predict_leaf("test_leaf.jpg"))