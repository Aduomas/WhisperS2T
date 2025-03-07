#!/usr/bin/env python3
import numpy as np
import tritonclient.http as httpclient
import argparse
import json
import os
import subprocess


def convert_audio_to_pcm(audio_path, sample_rate=16000):
    """Convert audio file to PCM float32 format."""
    temp_file = f"{audio_path}.temp.pcm"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "f32le",
        temp_file,
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with open(temp_file, "rb") as f:
        pcm_data = f.read()

    os.remove(temp_file)
    return pcm_data


def transcribe_audio(
    triton_url,
    audio_path,
    language="en",
    task="transcribe",
    word_timestamps=True,
    initial_prompt=None,
):
    """Transcribe audio using Triton Inference Server."""

    # Convert audio to PCM format
    pcm_data = convert_audio_to_pcm(audio_path)
    audio_data = np.frombuffer(pcm_data, dtype=np.float32)

    # Create Triton client
    client = httpclient.InferenceServerClient(url=triton_url)

    # Create the inference inputs
    inputs = []

    # Audio content
    audio_input = httpclient.InferInput("AUDIO_CONTENT", audio_data.shape, "FP32")
    audio_input.set_data_from_numpy(audio_data)
    inputs.append(audio_input)

    # Language code
    language_data = np.array([language], dtype=np.object_)
    language_input = httpclient.InferInput("LANGUAGE_CODE", [1], "BYTES")
    language_input.set_data_from_numpy(language_data)
    inputs.append(language_input)

    # Task
    task_data = np.array([task], dtype=np.object_)
    task_input = httpclient.InferInput("TASK", [1], "BYTES")
    task_input.set_data_from_numpy(task_data)
    inputs.append(task_input)

    # Word timestamps
    word_timestamps_data = np.array([word_timestamps], dtype=bool)
    word_timestamps_input = httpclient.InferInput("WORD_TIMESTAMPS", [1], "BOOL")
    word_timestamps_input.set_data_from_numpy(word_timestamps_data)
    inputs.append(word_timestamps_input)

    # Initial prompt (if provided)
    if initial_prompt:
        prompt_data = np.array([initial_prompt], dtype=np.object_)
        prompt_input = httpclient.InferInput("INITIAL_PROMPT", [1], "BYTES")
        prompt_input.set_data_from_numpy(prompt_data)
        inputs.append(prompt_input)

    # Define the outputs
    outputs = [
        httpclient.InferRequestedOutput("TRANSCRIPTION"),
        httpclient.InferRequestedOutput("WORD_TIMESTAMPS"),
    ]

    # Send request to Triton
    response = client.infer(model_name="whisper_asr", inputs=inputs, outputs=outputs)

    # Process the results
    transcription = response.as_numpy("TRANSCRIPTION")[0].decode("utf-8")
    word_timestamps_json = response.as_numpy("WORD_TIMESTAMPS")[0].decode("utf-8")
    word_info = json.loads(word_timestamps_json) if word_timestamps_json else []

    return {"transcription": transcription, "words": word_info}


def main():
    parser = argparse.ArgumentParser(description="Whisper ASR Triton Client")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument(
        "--language", default="en", choices=["en", "de", "fr"], help="Language code"
    )
    parser.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"], help="Task"
    )
    parser.add_argument(
        "--no-word-timestamps", action="store_true", help="Disable word timestamps"
    )
    parser.add_argument("--initial-prompt", help="Initial prompt for transcription")

    args = parser.parse_args()

    result = transcribe_audio(
        args.url,
        args.audio_path,
        args.language,
        args.task,
        not args.no_word_timestamps,
        args.initial_prompt,
    )

    print("Transcription:")
    print(result["transcription"])

    if not args.no_word_timestamps and result["words"]:
        print("\nWord Timestamps:")
        for word in result["words"]:
            print(f"{word['word']:<20} {word['start']:.2f} - {word['end']:.2f}")

    # Save result to file
    output_file = f"{os.path.splitext(args.audio_path)[0]}_transcription.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFull result saved to: {output_file}")


if __name__ == "__main__":
    main()
