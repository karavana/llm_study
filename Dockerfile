FROM python:3.10-slim

# System requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Needed files
COPY . .

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI port
EXPOSE 8000

# Default cmd
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
