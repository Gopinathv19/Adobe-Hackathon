FROM --platform=linux/amd64 python:3.10

WORKDIR /app

COPY . .

# Install torch & torchvision from PyTorch's official wheel index (CPU version)
RUN pip install --no-cache-dir torch  --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt


# Run the main script
CMD ["python","build.py", "process_pdfs.py"]
