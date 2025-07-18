import os
import sys
import wave
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_file_path = self._get_audio_file_path()
        self.wave_file = None
        self.recording_frames = 0

    def _audio_callback(self, indata, frames, time, status):
        if status:
            click.echo(f"⚠️  Audio warning: {status}", err=True)
        if self.wave_file:
            audio_int16 = (indata * 32767).astype(np.int16)
            self.wave_file.writeframes(audio_int16.tobytes())
            self.recording_frames += frames

    def record(self) -> Path:
        self._validate_audio_device()
        self._prepare_wave_file()

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate, channels=self.channels, callback=self._audio_callback, dtype=np.float32
            )
        except Exception as e:
            self._close_wave_file()
            raise RuntimeError(f"Failed to initialize audio stream: {e}")

        try:
            with stream:
                click.echo("🎤 Recording... Press Enter to stop", nl=False)
                input()
        except KeyboardInterrupt:
            click.echo("\n⏹️  Recording cancelled")
            self._close_wave_file()
            sys.exit(0)
        finally:
            self._close_wave_file()

        if self.recording_frames == 0:
            raise ValueError("No audio recorded")

        return self.audio_file_path

    def _validate_audio_device(self):
        try:
            default_input = sd.query_devices(kind="input")
            if default_input is None:
                raise RuntimeError("No audio input device found")
        except Exception as e:
            raise RuntimeError(f"Failed to access audio devices: {e}")

    def _get_audio_file_path(self) -> Path:
        if sys.platform == "win32":
            cache_dir = Path.home() / "AppData" / "Local" / "hns" / "Cache"
        elif sys.platform == "darwin":
            cache_dir = Path.home() / "Library" / "Caches" / "hns"
        else:
            cache_dir = Path.home() / ".cache" / "hns"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "last_recording.wav"

    def _prepare_wave_file(self):
        self.recording_frames = 0
        self.wave_file = wave.open(str(self.audio_file_path), "wb")
        self.wave_file.setnchannels(self.channels)
        self.wave_file.setsampwidth(2)  # 16-bit audio
        self.wave_file.setframerate(self.sample_rate)

    def _close_wave_file(self):
        if self.wave_file:
            self.wave_file.close()
            self.wave_file = None


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

    def transcribe(self, audio_source: Union[Path, str], stream_output: bool = False) -> str:
        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        }

        if self.language:
            transcribe_kwargs["language"] = self.language

        try:
            segments, _ = self.model.transcribe(str(audio_source), **transcribe_kwargs)

            transcription_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    if stream_output:
                        click.echo(text, nl=False)
                    transcription_parts.append(text)

            if stream_output and transcription_parts:
                click.echo()  # New line after streaming

            full_transcription = " ".join(transcription_parts)
            if not full_transcription:
                raise ValueError("No speech detected in audio")

            return full_transcription
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

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
@click.option("--last", is_flag=True, help="Transcribe the last recorded audio file")
def main(sample_rate: int, channels: int, list_models: bool, language: Optional[str], last: bool):
    """Record audio from microphone, transcribe it, and copy to clipboard."""

    if list_models:
        WhisperTranscriber.list_models()
        return

    try:
        if last:
            recorder = AudioRecorder(sample_rate, channels)
            audio_file_path = recorder._get_audio_file_path()
            if not audio_file_path.exists():
                click.echo(
                    "❌ No previous recording found. Record audio first by running 'hns' without --last flag.", err=True
                )
                sys.exit(1)
            click.echo("🔄 Transcribing last recording...")
        else:
            recorder = AudioRecorder(sample_rate, channels)
            audio_file_path = recorder.record()
            click.echo("🔄 Transcribing audio...")

        transcriber = WhisperTranscriber(language=language)
        transcription = transcriber.transcribe(audio_file_path)

        copy_to_clipboard(transcription)

    except (RuntimeError, ValueError) as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
