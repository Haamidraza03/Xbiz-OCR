FROM python:3.10-slim

WORKDIR /app

# Install small set of system deps required by OpenCV/Pillow, ffmpeg for media
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

# Copy app sources
COPY . .

# Expose default Flask port (change if needed)
EXPOSE 5000

# Start your app
CMD ["python", "main.py"]
