# Dockerfile.cpu - CPU build for PaddleOCR app
# Based on python:3.11-slim for good compatibility with paddlepaddle wheels

FROM python:3.11-slim

WORKDIR /app

# Install system packages required for OpenCV, image handling, building wheels if needed
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libsndfile1 \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .


RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt


# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "paddlewithbound.py"]
