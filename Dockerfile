# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for dlib & OpenCV
RUN apt-get update && apt-get install -y \
    build-essential cmake gfortran git wget curl \
    libsm6 libxext6 libxrender-dev \
    libgtk2.0-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]
