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

# Copy models from least likely to change to most likely to change
COPY YOLOmodels/DMbarcode.pt /app/YOLOmodels/DMbarcode.pt

COPY YOLOmodels/classifiers/BAR.pt /app/YOLOmodels/classifiers/BAR.pt
COPY YOLOmodels/classifiers/BOTTLE.pt /app/YOLOmodels/classifiers/BOTTLE.pt
COPY YOLOmodels/classifiers/CAN.pt /app/YOLOmodels/classifiers/CAN.pt
COPY YOLOmodels/classifiers/CHIPS.pt /app/YOLOmodels/classifiers/CHIPS.pt
COPY YOLOmodels/classifiers/CUP.pt /app/YOLOmodels/classifiers/CUP.pt
COPY YOLOmodels/classifiers/OTHER.pt /app/YOLOmodels/classifiers/OTHER.pt
COPY YOLOmodels/classifiers/SANDWICH.pt /app/YOLOmodels/classifiers/SANDWICH.pt
COPY YOLOmodels/classifiers/SPOONFUL.pt /app/YOLOmodels/classifiers/SPOONFUL.pt
COPY YOLOmodels/classifiers/TF-BLUE.pt /app/YOLOmodels/classifiers/TF-BLUE.pt
COPY YOLOmodels/classifiers/TF-BROWN.pt /app/YOLOmodels/classifiers/TF-BROWN.pt

COPY YOLOmodels/general_model.pt /app/YOLOmodels/general_model.pt

# Copy product list
COPY product_list.xlsx /app/product_list.xlsx

# Copy full pipeline
COPY FULL_PIPELINE.py /app/FULL_PIPELINE.py
COPY process.py /app/process.py

# Used previously with web app
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["python3", "-u", "process.py"]