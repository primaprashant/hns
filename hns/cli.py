import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pyperclip
import sounddevice as sd
from rich.console import Console

from .transcribers import create_transcriber, list_all_models

console = Console(stderr=True)
stdout_console = Console()


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


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_file_path = self._get_audio_file_path()
        self.wave_file = None
        self.recording_frames = 0

    def _audio_callback(self, indata, frames, time, status):
        if status:
            console.print(f"⚠️ [bold yellow]Audio warning: {status}[/bold yellow]")
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

        # Setup timer for live recording display
        start_time = time.time()
        recording_stopped = threading.Event()

        try:
            with stream:

                def update_timer():
                    """Update the display with elapsed time."""
                    while not recording_stopped.is_set():
                        elapsed = time.time() - start_time
                        time_str = format_duration(elapsed)
                        # Overwrite the same line
                        console.print(
                            f"🎤 [bold blue]Recording ...... {time_str} Press Enter to stop[/bold blue]", end="\r"
                        )
                        time.sleep(1)

                # Start timer thread
                timer_thread = threading.Thread(target=update_timer)
                timer_thread.daemon = True
                timer_thread.start()

                # Wait for user input
                input()

                # Stop timer and wait for thread to finish
                recording_stopped.set()
                timer_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish

                # Clear the recording line completely
                console.print(" " * 50, end="\r")  # Clear line with spaces

        except KeyboardInterrupt:
            recording_stopped.set()
            console.print("\n⏹️ [bold yellow]Recording cancelled[/bold yellow]")
            self._close_wave_file()
            sys.exit(0)
        finally:
            recording_stopped.set()
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


def copy_to_clipboard(text: str):
    pyperclip.copy(text)
    console.print("✅ [bold green]Copied to clipboard![/bold green]")


@click.command()
@click.option("--sample-rate", default=16000, help="Sample rate for audio recording")
@click.option("--channels", default=1, help="Number of audio channels")
@click.option("--list-models", is_flag=True, help="List available Whisper models and exit")
@click.option("--language", help="Force language detection (e.g., en, es, fr). Can also use HNS_LANG env var")
@click.option("--last", is_flag=True, help="Transcribe the last recorded audio file")
def main(sample_rate: int, channels: int, list_models: bool, language: Optional[str], last: bool):
    """Record audio from microphone, transcribe it, and copy to clipboard."""

    if list_models:
        list_all_models()
        return

    try:
        if last:
            recorder = AudioRecorder(sample_rate, channels)
            audio_file_path = recorder._get_audio_file_path()
            if not audio_file_path.exists():
                console.print(
                    "❌ [bold red]No previous recording found. Record audio first by running 'hns' without --last flag.[/bold red]"
                )
                sys.exit(1)
        else:
            recorder = AudioRecorder(sample_rate, channels)
            audio_file_path = recorder.record()
        transcriber = create_transcriber(language=language)
        transcription, _ = transcriber.transcribe(audio_file_path, show_progress=True)

        try:
            copy_to_clipboard(transcription)
        except Exception as e:
            console.print(f"⚠️ [bold yellow]Failed to copy to clipboard: {e}[/bold yellow]")

        stdout_console.print(transcription)

    except (RuntimeError, ValueError) as e:
        console.print(f"❌ [bold red]{e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
