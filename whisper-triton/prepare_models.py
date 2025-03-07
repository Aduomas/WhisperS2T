#!/usr/bin/env python3
import os
import sys
import whisper_s2t
import argparse


def prepare_models():
    """Download and prepare WhisperS2T models for TensorRT-LLM."""

    print("Preparing models for TensorRT-LLM...")

    models_dir = "/workspace/whisper_models"
    os.makedirs(models_dir, exist_ok=True)

    # Model identifiers
    model_identifiers = [
        "large-v3",  # English model
        "large-v3-german",  # German model
        "large-v3-french",  # French model
    ]

    for model_id in model_identifiers:
        print(f"Preparing model: {model_id}")
        try:
            # Load and prepare the model
            # This will download the model and convert it for TensorRT-LLM
            model = whisper_s2t.load_model(
                model_identifier=model_id,
                backend="TensorRT-LLM",
                asr_options={"word_timestamps": True},
            )
            print(f"Successfully prepared model: {model_id}")
        except Exception as e:
            print(f"Error preparing model {model_id}: {str(e)}")
            sys.exit(1)

    print("All models prepared successfully!")


if __name__ == "__main__":
    prepare_models()
