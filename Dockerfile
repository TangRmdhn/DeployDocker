# Gunakan image Python resmi sebagai base image
FROM python:3.10-slim

# Tetapkan working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke working directory
COPY requirements.txt .

# Instal semua library yang dibutuhkan dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek ke dalam container
# Ini akan menyalin app.py, model, dan folder templates
COPY . .

# Muat model di awal, ini akan membantu mengurangi waktu startup
# Anda harus memuat model dengan TensorFlow saat container dimulai
# Model akan dimuat di startup.sh

# Definisikan port yang akan di-expose oleh container
EXPOSE 8000

# Perintah untuk menjalankan server FastAPI dengan Uvicorn
# 'app:app' mengacu pada objek 'app' di dalam file 'app.py'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
