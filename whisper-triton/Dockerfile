ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_TAG=25.02-py3

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace
ENTRYPOINT []
SHELL ["/bin/bash", "-c"]

# Install essential packages first
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy TensorRT installation script
COPY ./install_tensorrt.sh /workspace/install_tensorrt.sh
RUN chmod +x /workspace/install_tensorrt.sh

# Run TensorRT installation
RUN /bin/bash /workspace/install_tensorrt.sh

# Install Jupyter and WhisperS2T
RUN pip3 install --no-cache-dir notebook jupyterlab ipywidgets && \
    pip3 install --no-cache-dir git+https://github.com/Aduomas/WhisperS2T.git

# Set up environment variables
RUN CUDNN_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))' 2>/dev/null || echo "") && \
    if [ ! -z "$CUDNN_PATH" ]; then echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${CUDNN_PATH}" >> ~/.bashrc; fi

# Copy model repository for Triton
COPY ./triton_model_repository /models

# Copy model preparation script
COPY ./prepare_models.py /workspace/prepare_models.py
RUN chmod +x /workspace/prepare_models.py

# Copy startup script and make it executable
COPY ./startup.sh /workspace/startup.sh
RUN chmod +x /workspace/startup.sh

# Set the startup script as the entry point
CMD ["/workspace/startup.sh"]