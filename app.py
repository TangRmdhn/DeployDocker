import os
import io
import tensorflow as tf
from tensorflow.keras.layers import Resizing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from fastapi import FastAPI, UploadFile, File
from starlette.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# --- 1. Konfigurasi Awal ---
# Definisikan label kelas CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Definisikan path ke model dan template secara dinamis
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'cifar10_final_model.keras'
TEMPLATE_PATH = BASE_DIR / 'templates' / 'index.html'

# --- 2. Inisialisasi Aplikasi FastAPI ---
app = FastAPI()

STATIC_PATH = BASE_DIR / 'static'
if not STATIC_PATH.exists():
    STATIC_PATH.mkdir()

app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# Mount folder 'static' untuk file statis jika ada (misalnya CSS, JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / 'static')), name="static")

# Variabel global untuk menyimpan model
model = None

# --- 3. Fungsi untuk Memuat Model saat Startup ---
@app.on_event("startup")
async def load_cifar10_model():
    """Load the TensorFlow model when the server starts."""
    global model
    print("✅ Loading model...")
    try:
        # Load the model and inform Keras about the custom Resizing layer
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={'Resizing': Resizing}
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None

# --- 4. Endpoint untuk Halaman Utama ---
@app.get("/", response_class=HTMLResponse)
async def serve_home_page():
    """Serve the index.html file for the web client."""
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>index.html not found!</h1>", status_code=500)
    
    with open(TEMPLATE_PATH, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --- 5. Endpoint untuk Prediksi Gambar ---
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model has not been loaded"}, 500

    try:
        # Read the file content
        contents = await file.read()
        
        # Preprocess the image
        img = load_img(io.BytesIO(contents), target_size=(32, 32))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        response = {
            "prediction": CLASS_NAMES[predicted_class_index],
            "confidence": confidence
        }
        
        return response
    
    except Exception as e:
        return {"error": str(e)}, 500
