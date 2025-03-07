ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_TAG=25.02-py3

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace
ENTRYPOINT []
SHELL ["/bin/bash", "-c"]

COPY ./install_tensorrt.sh /workspace/install_tensorrt.sh

RUN apt update && apt-get install -y libsndfile1 ffmpeg git && \
    pip3 install --no-cache-dir notebook jupyterlab ipywidgets && \
    pip3 install --no-cache-dir git+https://github.com/Aduomas/WhisperS2T.git && \
    CUDNN_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'${CUDNN_PATH} >> ~/.bashrc

RUN /bin/bash /workspace/install_tensorrt.sh

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./triton_model_repository /models
COPY ./prepare_models.py /workspace/prepare_models.py

CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]