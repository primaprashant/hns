import os
import sys
from typing import List, Optional

import click
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_data: List[np.ndarray] = []

    def _audio_callback(self, indata, frames, time, status):
        if status:
            click.echo(f"⚠️  Audio warning: {status}", err=True)
        self.audio_data.append(indata.copy())

    def record(self) -> np.ndarray:
        self._validate_audio_device()

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate, channels=self.channels, callback=self._audio_callback, dtype=np.float32
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio stream: {e}")

        try:
            with stream:
                click.echo("🎤 Recording... Press Enter to stop")
                input()
        except KeyboardInterrupt:
            click.echo("\n⏹️  Recording cancelled")
            sys.exit(0)

        if not self.audio_data:
            raise ValueError("No audio recorded")

        audio_array = np.concatenate(self.audio_data, axis=0)
        return audio_array.flatten() if audio_array.ndim > 1 else audio_array

    def _validate_audio_device(self):
        try:
            default_input = sd.query_devices(kind="input")
            if default_input is None:
                raise RuntimeError("No audio input device found")
        except Exception as e:
            raise RuntimeError(f"Failed to access audio devices: {e}")


class WhisperTranscriber:
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

    def __init__(self, model_name: Optional[str] = None, language: Optional[str] = None):
        self.model_name = self._get_model_name(model_name)
        self.language = language or os.environ.get("HNS_LANG")
        self.model = self._load_model()

    def _get_model_name(self, model_name: Optional[str]) -> str:
        model = model_name or os.environ.get("HNS_WHISPER_MODEL", "base")

        if model not in self.VALID_MODELS:
            click.echo(f"⚠️  Invalid model '{model}', using 'base' instead", err=True)
            click.echo(f"    Available models: {', '.join(self.VALID_MODELS)}")
            return "base"

        return model

    def _load_model(self) -> WhisperModel:
        try:
            return WhisperModel(self.model_name, device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def transcribe(self, audio_array: np.ndarray) -> str:
        normalized_audio = self._normalize_audio(audio_array)

        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        }

        if self.language:
            transcribe_kwargs["language"] = self.language

        try:
            segments, _ = self.model.transcribe(normalized_audio, **transcribe_kwargs)
            transcription = " ".join(segment.text.strip() for segment in segments)

            if not transcription:
                raise ValueError("No speech detected in audio")

            return transcription
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        return audio_array / (np.max(np.abs(audio_array)) + 1e-7)

    @classmethod
    def list_models(cls):
        click.echo("Available Whisper models:")
        for model in cls.VALID_MODELS:
            click.echo(f"  • {model}")
        click.echo("\nEnvironment variables:")
        click.echo("  export HNS_WHISPER_MODEL=<model-name>")
        click.echo("  export HNS_LANG=<language-code>  # e.g., en, es, fr")


def copy_to_clipboard(text: str):
    pyperclip.copy(text)
    click.echo("✅ Transcription copied to clipboard!")
    click.echo(f"\n{text}")


@click.command()
@click.option("--sample-rate", default=16000, help="Sample rate for audio recording")
@click.option("--channels", default=1, help="Number of audio channels")
@click.option("--list-models", is_flag=True, help="List available Whisper models and exit")
@click.option("--language", help="Force language detection (e.g., en, es, fr). Can also use HNS_LANG env var")
def main(sample_rate: int, channels: int, list_models: bool, language: Optional[str]):
    """Record audio from microphone, transcribe it, and copy to clipboard."""

    if list_models:
        WhisperTranscriber.list_models()
        return

    try:
        recorder = AudioRecorder(sample_rate, channels)
        audio_array = recorder.record()

        transcriber = WhisperTranscriber(language=language)
        transcription = transcriber.transcribe(audio_array)

        copy_to_clipboard(transcription)

    except (RuntimeError, ValueError) as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
