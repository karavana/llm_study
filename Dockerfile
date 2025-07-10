FROM python:3.10-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Gereken dosyalar
COPY . .

# Python bağımlılıkları
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI sunucusu için port
EXPOSE 8000

# Varsayılan komut
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
