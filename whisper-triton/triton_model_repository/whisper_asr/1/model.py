import json
import numpy as np
import triton_python_backend_utils as pb_utils
import whisper_s2t
import logging
import tempfile
import os
import time
import traceback


class TritonPythonModel:
    """Python model for Whisper ASR using WhisperS2T."""

    def initialize(self, args):
        """Initialize the model.
        Args:
            args: Dict containing model configuration parameters
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("whisper_asr")
        self.logger.info("Initializing Whisper ASR model")

        # Debug info
        self.logger.info(f"Model config: {args}")

        # Initialize empty models dictionary (will load on demand)
        self.models = {}
        self.logger.info("Models will be loaded on first request")

        # Store model configuration
        self.model_config = args["model_config"]
        self.logger.info("Initialization complete")

    def _load_model(self, language_code):
        """Load a model for the specified language code.

        Args:
            language_code: The language code to load a model for

        Returns:
            The loaded model or None if loading failed
        """
        try:
            # Determine model identifier based on language
            if language_code == "en":
                model_id = "large-v3"
            elif language_code == "de":
                model_id = "large-v3-german"
            elif language_code == "fr":
                model_id = "large-v3-french"
            else:
                # Default to English for unsupported languages
                self.logger.warning(
                    f"Unsupported language code: {language_code}, using English model"
                )
                language_code = "en"
                model_id = "large-v3"

            self.logger.info(f"Loading model: {model_id} for language: {language_code}")

            # Load the model
            model = whisper_s2t.load_model(
                model_identifier=model_id,
                backend="TensorRT-LLM",
                asr_options={"word_timestamps": True},
            )

            self.logger.info(f"Successfully loaded model for {language_code}")
            return model

        except Exception as e:
            self.logger.error(f"Error loading model for {language_code}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def execute(self, requests):
        """Process the inference requests.
        Args:
            requests: List of pb_utils.InferenceRequest objects
        Returns:
            List of pb_utils.InferenceResponse objects
        """
        responses = []

        for request in requests:
            try:
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

                # Get optional parameters
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
                with tempfile.NamedTemporaryFile(
                    suffix=".pcm", delete=False
                ) as temp_file:
                    temp_filename = temp_file.name
                    temp_file.write(audio_content.tobytes())

                # Load model on demand if not already loaded
                if language_code not in self.models:
                    self.logger.info(
                        f"Model for {language_code} not loaded yet, loading..."
                    )
                    self.models[language_code] = self._load_model(language_code)

                # Fall back to English if requested model couldn't be loaded
                if self.models.get(language_code) is None:
                    self.logger.warning(
                        f"No model available for {language_code}, falling back to English"
                    )

                    # Try loading English model if not already loaded
                    if "en" not in self.models:
                        self.models["en"] = self._load_model("en")

                    if self.models.get("en") is None:
                        # Failed to load any model
                        error = pb_utils.TritonError(
                            "Failed to load any speech recognition model"
                        )
                        responses.append(pb_utils.InferenceResponse(error=error))
                        continue

                    language_code = "en"

                # Get the model to use
                model = self.models[language_code]

                # Transcribe the audio
                self.logger.info(
                    f"Transcribing audio with language: {language_code}, task: {task}"
                )
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
                if word_timestamps and segments and "words" in segments[0]:
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
                self.logger.error(traceback.format_exc())
                # Return an error response
                error = pb_utils.TritonError(f"Transcription error: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))

            # Clean up temporary file
            try:
                if "temp_filename" in locals():
                    os.unlink(temp_filename)
            except:
                pass

        return responses

    def finalize(self):
        """Clean up resources when the model is unloaded."""
        self.logger.info("Finalizing Whisper ASR model")
        # Clear the models
        self.models.clear()
