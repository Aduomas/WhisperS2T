import json
import numpy as np
import triton_python_backend_utils as pb_utils
import whisper_s2t
import logging
import tempfile
import os
import time


class TritonPythonModel:
    """Python model for Whisper ASR using WhisperS2T."""

    def initialize(self, args):
        """Initialize the model.
        Args:
            args: Dict containing model configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Whisper ASR model")

        # Load the WhisperS2T models with TensorRT-LLM backend
        self.models = {
            "en": whisper_s2t.load_model(
                model_identifier="large-v3",
                backend="TensorRT-LLM",
                asr_options={"word_timestamps": True},
            ),
            "de": whisper_s2t.load_model(
                model_identifier="large-v3-german",
                backend="TensorRT-LLM",
                asr_options={"word_timestamps": True},
            ),
            "fr": whisper_s2t.load_model(
                model_identifier="large-v3-french",
                backend="TensorRT-LLM",
                asr_options={"word_timestamps": True},
            ),
        }

        self.logger.info("Whisper ASR models loaded successfully")

    def execute(self, requests):
        """Process the inference requests.
        Args:
            requests: List of pb_utils.InferenceRequest objects
        Returns:
            List of pb_utils.InferenceResponse objects
        """
        responses = []

        for request in requests:
            # Get input tensors
            audio_content = pb_utils.get_input_tensor_by_name(
                request, "AUDIO_CONTENT"
            ).as_numpy()
            language_code = (
                pb_utils.get_input_tensor_by_name(request, "LANGUAGE_CODE")
                .as_numpy()[0]
                .decode("utf-8")
            )
            task = (
                pb_utils.get_input_tensor_by_name(request, "TASK")
                .as_numpy()[0]
                .decode("utf-8")
            )

            try:
                word_timestamps_tensor = pb_utils.get_input_tensor_by_name(
                    request, "WORD_TIMESTAMPS"
                )
                word_timestamps = bool(word_timestamps_tensor.as_numpy()[0])
            except:
                word_timestamps = True

            try:
                initial_prompt_tensor = pb_utils.get_input_tensor_by_name(
                    request, "INITIAL_PROMPT"
                )
                initial_prompt = initial_prompt_tensor.as_numpy()[0].decode("utf-8")
                if not initial_prompt:
                    initial_prompt = None
            except:
                initial_prompt = None

            # Save audio content to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_content.tobytes())

            # Select the model based on language
            if language_code in self.models:
                model = self.models[language_code]
            else:
                # Default to English model
                model = self.models["en"]
                language_code = "en"

            try:
                # Transcribe the audio
                result = model.transcribe_with_vad(
                    [temp_filename],
                    lang_codes=[language_code],
                    tasks=[task],
                    initial_prompts=[initial_prompt],
                    batch_size=16,
                )

                # Process the transcription result
                segments = result[0]

                # Combine all segments into one result
                full_text = " ".join([segment["text"].strip() for segment in segments])

                # Get word timestamps if available and requested
                word_info = []
                if word_timestamps and "words" in segments[0]:
                    for segment in segments:
                        if "words" in segment:
                            word_info.extend(segment["words"])

                # Create response tensors
                transcription_tensor = pb_utils.Tensor(
                    "TRANSCRIPTION", np.array([full_text], dtype=np.object_)
                )

                word_timestamps_tensor = pb_utils.Tensor(
                    "WORD_TIMESTAMPS",
                    np.array([json.dumps(word_info)], dtype=np.object_),
                )

                # Create and append the inference response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[transcription_tensor, word_timestamps_tensor]
                )
                responses.append(inference_response)

            except Exception as e:
                self.logger.error(f"Error during transcription: {str(e)}")
                # Return an error response
                error = pb_utils.TritonError(f"Transcription error: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))

            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass

        return responses

    def finalize(self):
        """Clean up resources when the model is unloaded."""
        self.logger.info("Finalizing Whisper ASR model")
