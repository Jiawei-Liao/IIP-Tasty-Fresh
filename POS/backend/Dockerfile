# NVIDIA CUDA image for Jetson Jetpack 5.1.1
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libdmtx0b \
    libsm6 \
    libxext6 \
    python3-opencv \
    nvidia-container-runtime \
    && rm -rf /var/lib/apt/lists/*
    
# Clone and build jetson-ffmpeg
RUN git clone https://github.com/jocover/jetson-ffmpeg.git /tmp/jetson-ffmpeg && \
    cd /tmp/jetson-ffmpeg && \
    mkdir build && cd build && \
    cmake .. -DJP4.6=ON -DCMAKE_INSTALL_PREFIX=/usr && \
    make -j$(nproc) && make install && \
    rm -rf /tmp/jetson-ffmpeg

# Set library paths
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Remove any explicit OpenCV installation from pip
RUN python3 -m pip uninstall -y opencv-python opencv-python-headless

# Copy rest of the application
COPY . .

# Used previously with web app
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["python3", "FULL_PIPELINE.py"]