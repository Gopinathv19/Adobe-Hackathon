# Use official Python image
FROM --platform=linux/amd64 python:3.10

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Default command (change to your main script as needed)
CMD ["python", "process_pdfs.py"]
