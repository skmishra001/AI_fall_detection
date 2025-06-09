from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("plant_disease_model.h5")

class_names = ["Apple___Scab", "Apple___Black_rot", "Apple___healthy", "Corn___Common_rust", "Tomato___Early_blight"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    result = class_names[index]
    return render_template("index.html", prediction_text=f"Disease Detected: {result}")

if __name__ == "__main__":
    app.run(debug=True)