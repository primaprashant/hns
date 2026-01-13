# #######################################
# Imports
# #######################################

import os
import time
import wave
from pathlib import Path
from typing import Optional, Union

from rich.console import Console

# #######################################
# Module State
# #######################################

console = Console(stderr=True)

# #######################################
# Utilities
# #######################################

def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS or MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


# #######################################
# WhisperTranscriber
# #######################################

class WhisperTranscriber:
    """Transcriber backend using faster-whisper for speech-to-text.

    Attributes:
        VALID_MODELS: List of supported Whisper model names.
        model_name: The name of the loaded Whisper model.
        language: Optional language code for transcription.
        model: The loaded WhisperModel instance.
    """

    # ##############################
    # Constants
    # ##############################
    VALID_MODELS = [
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-medium.en",
        "distil-small.en",
        "distil-large-v3",
        "distil-large-v3.5",
        "large-v3-turbo",
        "turbo",
    ]

    # ##############################
    # Initialization
    # ##############################

    def __init__(self, model_name: Optional[str] = None, language: Optional[str] = None):
        """Initialize the WhisperTranscriber.

        Args:
            model_name: Name of the Whisper model to use. Defaults to 'base'
                or the value of HNS_WHISPER_MODEL environment variable.
            language: Language code for transcription (e.g., 'en', 'es').
                Defaults to the value of HNS_LANG environment variable.
        """
        self.model_name = self._get_model_name(model_name)
        self.language = language or os.environ.get("HNS_LANG")
        self.model = self._load_model()

    # ##############################
    # Model Configuration
    # ##############################

    def _get_model_name(self, model_name: Optional[str]) -> str:
        """Resolve and validate the model name.

        Args:
            model_name: Requested model name, or None to use default/env var.

        Returns:
            A valid model name. Falls back to 'base' if invalid.
        """
        model = model_name or os.environ.get("HNS_WHISPER_MODEL", "base")

        if model not in self.VALID_MODELS:
            console.print(f"[bold yellow]Invalid model '{model}', using 'base' instead[/bold yellow]")
            console.print(f"    [dim]Available models: {', '.join(self.VALID_MODELS)}[/dim]")
            return "base"

        return model

    def _load_model(self):
        """Load the Whisper model.

        Returns:
            A WhisperModel instance configured for CPU inference.

        Raises:
            RuntimeError: If the model fails to load.
        """
        from faster_whisper import WhisperModel

        try:
            return WhisperModel(self.model_name, device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # ##############################
    # Audio Utilities
    # ##############################

    def _get_audio_duration(self, audio_file_path: Union[Path, str]) -> Optional[float]:
        """Get duration of audio file in seconds.

        Args:
            audio_file_path: Path to the WAV audio file.

        Returns:
            Duration in seconds, or None if the file cannot be read.
        """
        try:
            with wave.open(str(audio_file_path), "rb") as audio_file:
                frames = audio_file.getnframes()
                sample_rate = audio_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception:
            return None

    # ##############################
    # Transcription
    # ##############################

    def transcribe(self, audio_source: Union[Path, str], show_progress: bool = True) -> tuple[str, Optional[float]]:
        """Transcribe audio file to text.

        Args:
            audio_source: Path to the audio file to transcribe.
            show_progress: If True, display a progress indicator during
                transcription. Defaults to True.

        Returns:
            A tuple of (transcription_text, elapsed_seconds). elapsed_seconds
            is None if show_progress is False.

        Raises:
            ValueError: If no speech is detected in the audio.
            RuntimeError: If transcription fails.
        """

        # ####################
        # Configure Transcription Parameters
        # ####################

        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        }

        if self.language:
            transcribe_kwargs["language"] = self.language

        # ####################
        # Perform Transcription
        # ####################

        try:
            start_time = time.time()

            if show_progress:
                # ##### Transcription with Progress Display
                import queue
                import threading

                progress_queue = queue.Queue()
                transcription_complete = threading.Event()

                def transcribe_worker():
                    """Worker function to perform transcription in background."""
                    try:
                        segments, _ = self.model.transcribe(str(audio_source), **transcribe_kwargs)
                        transcription_parts = []
                        for segment in segments:
                            text = segment.text.strip()
                            if text:
                                transcription_parts.append(text)
                        progress_queue.put(("result", transcription_parts))
                    except Exception as e:
                        progress_queue.put(("error", e))
                    finally:
                        transcription_complete.set()

                # Start transcription in background
                worker_thread = threading.Thread(target=transcribe_worker)
                worker_thread.daemon = True
                worker_thread.start()

                # Display progress with elapsed timer
                while not transcription_complete.is_set():
                    elapsed = time.time() - start_time
                    time_str = format_duration(elapsed)
                    console.print(f"[bold blue]Transcribing ... {time_str}[/bold blue]", end="\r")
                    time.sleep(1)

                # Print a new line
                console.print("")

                # Retrieve result from worker
                result_type, result_data = progress_queue.get()
                if result_type == "error":
                    raise result_data
                transcription_parts = result_data
            else:
                # ##### Transcription without Progress Display
                segments, _ = self.model.transcribe(str(audio_source), **transcribe_kwargs)
                transcription_parts = []
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        transcription_parts.append(text)

            # ####################
            # Assemble and Validate Result
            # ####################

            full_transcription = " ".join(transcription_parts)
            if not full_transcription:
                raise ValueError("No speech detected in audio")

            elapsed_total = time.time() - start_time
            return full_transcription, elapsed_total if show_progress else None
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    # ##############################
    # Class Methods
    # ##############################

    @classmethod
    def list_models(cls) -> None:
        """Print available Whisper models to the console."""
        console.print("[bold cyan]Available Whisper models:[/bold cyan]")
        for model in cls.VALID_MODELS:
            console.print(f"  [dim]{model}[/dim]")
        console.print("\n[bold cyan]Environment variables:[/bold cyan]")
        console.print("  [dim]export HNS_WHISPER_MODEL=<model-name>[/dim]")
        console.print("  [dim]export HNS_LANG=<language-code>  # e.g., en, es, fr[/dim]")
