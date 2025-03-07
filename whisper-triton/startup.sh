#!/bin/bash
set -e

echo "==== Setting up environment ===="
source ~/.bashrc

echo "==== Checking model repository structure ===="
ls -la /models

echo "==== Verifying WhisperS2T installation ===="
python3 -c "import whisper_s2t; print('WhisperS2T version/path:', whisper_s2t.__file__)"

echo "==== Checking for existing models ===="
find /workspace -name "*.bin" | grep -i whisper || echo "No model files found yet"

echo "==== Preparing Whisper models (this may take a while) ===="
python3 /workspace/prepare_models.py

echo "==== Starting Triton Inference Server ===="
exec tritonserver --model-repository=/models --log-verbose=1