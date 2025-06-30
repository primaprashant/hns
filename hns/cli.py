import os
import sys
import threading
from typing import Optional

import click
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel

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


@click.command()
@click.option("--sample-rate", default=16000, help="Sample rate for audio recording")
@click.option("--channels", default=1, help="Number of audio channels")
@click.option("--list-models", is_flag=True, help="List available Whisper models and exit")
@click.option("--language", default=None, help="Force language detection (e.g., en, es, fr). Can also use HNS_LANG env var")
def main(sample_rate: int, channels: int, list_models: bool, language: Optional[str]) -> None:
    """Record audio from microphone, transcribe it, and copy to clipboard."""

    if list_models:
        click.echo("Available Whisper models:")
        for model in VALID_MODELS:
            click.echo(f"  • {model}")
        click.echo("\nEnvironment variables:")
        click.echo("  export HNS_WHISPER_MODEL=<model-name>")
        click.echo("  export HNS_LANG=<language-code>  # e.g., en, es, fr")
        sys.exit(0)
    try:
        # Check audio device availability
        try:
            default_input = sd.query_devices(kind="input")
            if default_input is None:
                click.echo("❌ No audio input device found", err=True)
                sys.exit(1)
        except Exception as e:
            click.echo(f"❌ Failed to access audio devices: {e}", err=True)
            sys.exit(1)
        # Initialize recording state
        audio_data = []
        stop_event = threading.Event()

        def audio_callback(indata, frames, time, status):
            if status:
                click.echo(f"⚠️  Audio warning: {status}", err=True)
            audio_data.append(indata.copy())

        # Start recording in a separate thread
        try:
            stream = sd.InputStream(
                samplerate=sample_rate, channels=channels, callback=audio_callback, dtype=np.float32
            )
        except Exception as e:
            click.echo(f"❌ Failed to initialize audio stream: {e}", err=True)
            sys.exit(1)

        try:
            with stream:
                click.echo("🎤 Recording... Press Enter to stop")
                # Wait for Enter key
                input()
                stop_event.set()
        except KeyboardInterrupt:
            click.echo("\n⏹️  Recording cancelled")
            sys.exit(0)
        except Exception as e:
            click.echo(f"❌ Recording error: {e}", err=True)
            sys.exit(1)

        # Concatenate audio data
        if not audio_data:
            click.echo("❌ No audio recorded", err=True)
            sys.exit(1)

        audio_array = np.concatenate(audio_data, axis=0)

        # Flatten to 1D array if multi-channel
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Get model name from environment variable
        model_name = os.environ.get("HNS_WHISPER_MODEL", "base")

        if model_name not in VALID_MODELS:
            click.echo(f"⚠️  Invalid model '{model_name}', using 'base' instead", err=True)
            click.echo(f"    Available models: {', '.join(VALID_MODELS)}")
            model_name = "base"

        # Initialize whisper model
        try:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as e:
            click.echo(f"❌ Failed to load model: {e}", err=True)
            sys.exit(1)

        # Transcribe audio

        # Ensure audio is properly normalized
        audio_array = audio_array / np.max(np.abs(audio_array) + 1e-7)  # Normalize to [-1, 1]

        try:
            # Let faster-whisper handle the audio format conversion
            transcribe_kwargs = {
                "beam_size": 5,
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500, speech_pad_ms=400, threshold=0.5),
            }

            # Add language if specified (CLI flag takes precedence over env var)
            if language:
                transcribe_kwargs["language"] = language
            elif (env_language := os.environ.get("HNS_LANG")):
                transcribe_kwargs["language"] = env_language
            segments, info = model.transcribe(audio_array, **transcribe_kwargs)
        except Exception as e:
            click.echo(f"❌ Transcription failed: {e}", err=True)
            sys.exit(1)

        # Collect transcription text
        transcription = " ".join(segment.text.strip() for segment in segments)

        if not transcription:
            click.echo("❌ No speech detected in audio", err=True)
            sys.exit(1)

        # Copy to clipboard
        pyperclip.copy(transcription)

        click.echo("✅ Transcription copied to clipboard!")
        click.echo(f"\n{transcription}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
