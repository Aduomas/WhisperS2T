#!/bin/bash
set -e

echo ""
echo "###########################[ Installing Build Tools ]##########################"
apt-get update && apt-get install -y build-essential ca-certificates ccache cmake gnupg2 wget curl gdb

echo ""
echo "###########################[ Installing OpenMPI ]###########################"
apt-get update && apt-get -y install openmpi-bin libopenmpi-dev

echo ""
echo "###########################[ Installing PyTorch for CUDA 12.6 ]###########################"
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "###########################[ Installing TensorRT-LLM ]###########################"
# Install the latest compatible version of TensorRT-LLM
pip3 install --no-cache-dir tensorrt-llm --extra-index-url https://pypi.nvidia.com

echo ""
echo "TensorRT-LLM installation completed"