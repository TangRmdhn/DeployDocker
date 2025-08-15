import os
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException

# Tentukan label kelas CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Tentukan path model
MODEL_PATH = 'cifar10_final_model.keras'

# Load model di sini. Kode ini akan dijalankan saat server dimulai.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat.")
except Exception as e:
    model = None
    print(f"Gagal memuat model: {e}")
    # Jika model gagal dimuat, kita akan tangani di endpoint

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Endpoint untuk prediksi gambar
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Cek apakah model berhasil dimuat
    if model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat")

    try:
        # Baca konten file
        contents = await file.read()
        
        # Preprocessing gambar
        img = load_img(io.BytesIO(contents), target_size=(32, 32))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        response = {
            "prediction": CLASS_NAMES[predicted_class_index],
            "confidence": confidence
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
