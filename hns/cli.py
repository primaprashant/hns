import os
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

console = Console()


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
                        console.print(f"🎤 [bold blue]Recording {time_str}... Press Enter to stop[/bold blue]", end="\r")
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

    def _get_audio_duration(self, audio_file_path: Union[Path, str]) -> Optional[float]:
        """Get duration of audio file in seconds."""
        try:
            with wave.open(str(audio_file_path), "rb") as audio_file:
                frames = audio_file.getnframes()
                sample_rate = audio_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception:
            return None

    def _get_model_name(self, model_name: Optional[str]) -> str:
        model = model_name or os.environ.get("HNS_WHISPER_MODEL", "base")

        if model not in self.VALID_MODELS:
            console.print(f"⚠️ [bold yellow]Invalid model '{model}', using 'base' instead[/bold yellow]")
            console.print(f"    [dim]Available models: {', '.join(self.VALID_MODELS)}[/dim]")
            return "base"

        return model

    def _load_model(self) -> WhisperModel:
        try:
            return WhisperModel(self.model_name, device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def transcribe(
        self, audio_source: Union[Path, str], stream_output: bool = False, show_progress: bool = True
    ) -> str:
        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        }

        if self.language:
            transcribe_kwargs["language"] = self.language

        try:
            audio_duration = self._get_audio_duration(audio_source) if show_progress else None

            if show_progress:
                if audio_duration:
                    duration_str = format_duration(audio_duration)
                    console.print(f"🔄 [bold blue]Transcribing {duration_str} of audio...[/bold blue]")
                else:
                    console.print("🔄 [bold blue]Transcribing audio...[/bold blue]")

            start_time = time.time()

            if show_progress:
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
                                if stream_output:
                                    click.echo(text, nl=False)
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

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Processing audio..."),
                    TimeElapsedColumn(),
                    transient=False,
                ) as progress:
                    task = progress.add_task("Analyzing speech patterns", total=None)

                    while not transcription_complete.is_set():
                        elapsed = time.time() - start_time
                        if audio_duration and elapsed > 0:
                            # Rough estimation: transcription usually takes 10-30% of audio duration
                            estimated_progress = min(95, (elapsed / (audio_duration * 0.2)) * 100)
                            progress.update(
                                task,
                                description=f"[bold blue]Processing (~{estimated_progress:.0f}% estimated)[/bold blue]",
                            )
                        else:
                            progress.update(
                                task, description=f"[bold blue]Processing ({elapsed:.1f}s elapsed)[/bold blue]"
                            )

                        time.sleep(0.1)  # Update every 100ms

                result_type, result_data = progress_queue.get()
                if result_type == "error":
                    raise result_data
                transcription_parts = result_data
            else:
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

            elapsed_total = time.time() - start_time
            return full_transcription, elapsed_total if show_progress else None
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    @classmethod
    def list_models(cls):
        console.print("ℹ️ [bold cyan]Available Whisper models:[/bold cyan]")
        for model in cls.VALID_MODELS:
            console.print(f"  • [dim]{model}[/dim]")
        console.print("\nℹ️ [bold cyan]Environment variables:[/bold cyan]")
        console.print("  [dim]export HNS_WHISPER_MODEL=<model-name>[/dim]")
        console.print("  [dim]export HNS_LANG=<language-code>  # e.g., en, es, fr[/dim]")


def copy_to_clipboard(text: str, elapsed_time: Optional[float] = None):
    pyperclip.copy(text)
    if elapsed_time:
        console.print(f"✅ [bold green]Transcribed and copied to clipboard in {elapsed_time:.1f}s![/bold green]")
    else:
        console.print("✅ [bold green]Transcribed and copied to clipboard![/bold green]")
    console.print(f"\n{text}")


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
                console.print(
                    "❌ [bold red]No previous recording found. Record audio first by running 'hns' without --last flag.[/bold red]"
                )
                sys.exit(1)
        else:
            recorder = AudioRecorder(sample_rate, channels)
            audio_file_path = recorder.record()
        transcriber = WhisperTranscriber(language=language)
        transcription, elapsed_time = transcriber.transcribe(audio_file_path, show_progress=True)

        copy_to_clipboard(transcription, elapsed_time)

    except (RuntimeError, ValueError) as e:
        console.print(f"❌ [bold red]{e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
