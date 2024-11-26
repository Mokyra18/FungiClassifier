from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os

# Inisialisasi Flask
app = Flask(__name__)

# Direktori untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = "static/upload"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Label klasifikasi
dic = {
    0: 'poisonous sporocarp',
    1: 'poisonous mushroom sporocarp',
    2: 'edible mushroom sporocarp',
    3: 'edible sporocarp',
}

# Memuat model
model = tf.keras.models.load_model('MobileNetV2-v1-Deteksi Jamur-75.0.h5')

def predict_label(img_path):
    # Memuat dan memproses gambar
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    return dic[predicted_class]

# Route utama
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

# Route untuk menangani pengunggahan dan prediksi
@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(img_path)
        
        # Lakukan prediksi
        prediction = predict_label(img_path)
        return render_template("classification.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
