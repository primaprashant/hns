import os
import sys
import time
import wave
from pathlib import Path

import click
import numpy as np
import pyperclip
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.text import Text

console = Console(stderr=True)
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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


class TimerLine:
    def __init__(self, template: str):
        self.template = template
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def __rich__(self) -> Text:
        spinner = SPINNER_FRAMES[int(self.elapsed * 10) % len(SPINNER_FRAMES)]
        return Text.from_markup(self.template.format(elapsed=format_duration(self.elapsed), spinner=spinner))


def live_line(line: TimerLine) -> Live:
    return Live(line, console=console, refresh_per_second=10, transient=True)


def get_audio_file_path() -> Path:
    cache_dir_override = os.environ.get("HNS_CACHE_DIR")
    if cache_dir_override:
        cache_dir = Path(cache_dir_override).expanduser()
    elif sys.platform == "win32":
        cache_dir = Path.home() / "AppData" / "Local" / "hns" / "Cache"
    elif sys.platform == "darwin":
        cache_dir = Path.home() / "Library" / "Caches" / "hns"
    else:
        cache_dir = Path.home() / ".cache" / "hns"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "last_recording.wav"


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_file_path = get_audio_file_path()
        self.wave_file = None
        self.recording_frames = 0

    @property
    def duration(self) -> float:
        return self.recording_frames / self.sample_rate

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            console.print(f"[yellow]![/yellow] audio warning: {escape(str(status))}")
        if self.wave_file:
            audio_int16 = (indata * 32767).astype(np.int16)
            self.wave_file.writeframes(audio_int16.tobytes())
            self.recording_frames += frames

    def record(self) -> Path:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "recording needs an interactive terminal; use --last to retranscribe the previous recording"
            )

        self._validate_audio_device()
        temp_path = self.audio_file_path.with_name(self.audio_file_path.name + ".tmp")
        self._prepare_wave_file(temp_path)

        try:
            import sounddevice as sd

            stream = sd.InputStream(
                samplerate=self.sample_rate, channels=self.channels, callback=self._audio_callback, dtype=np.float32
            )
        except Exception as e:
            self._close_wave_file()
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"failed to initialize audio stream: {e}") from e

        timer = TimerLine("[red]●[/red] recording {elapsed} [dim]· press enter to stop[/dim]")
        try:
            with stream, live_line(timer):
                input()
        except KeyboardInterrupt:
            self._close_wave_file()
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            self._close_wave_file()

        if self.recording_frames == 0:
            temp_path.unlink(missing_ok=True)
            raise ValueError("no audio recorded")

        temp_path.replace(self.audio_file_path)
        return self.audio_file_path

    def _validate_audio_device(self):
        try:
            import sounddevice as sd

            sd.query_devices(kind="input")
        except Exception as e:
            raise RuntimeError(f"no audio input device available: {e}") from e

    def _prepare_wave_file(self, path: Path):
        self.recording_frames = 0
        self.wave_file = wave.open(str(path), "wb")
        self.wave_file.setnchannels(self.channels)
        self.wave_file.setsampwidth(2)
        self.wave_file.setframerate(self.sample_rate)

    def _close_wave_file(self):
        if self.wave_file:
            self.wave_file.close()
            self.wave_file = None


class WhisperTranscriber:
    VALID_MODELS = (
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
    )

    def __init__(self, model_name: str | None = None, language: str | None = None):
        self.model_name = self._get_model_name(model_name)
        self.language = language or os.environ.get("HNS_LANG")
        self.model = self._load_model()

    def _get_model_name(self, model_name: str | None) -> str:
        model = model_name or os.environ.get("HNS_WHISPER_MODEL", "base")

        if model not in self.VALID_MODELS:
            console.print(f"[yellow]![/yellow] invalid model '{escape(model)}', using 'base'")
            console.print("  [dim]run hns --list-models to see available models[/dim]")
            return "base"

        return model

    def _load_model(self):
        from faster_whisper import WhisperModel

        timer = TimerLine(f"[cyan]{{spinner}}[/cyan] loading {self.model_name} model")
        try:
            with live_line(timer):
                return WhisperModel(self.model_name, device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(f"failed to load model '{self.model_name}': {e}") from e

    def transcribe(self, audio_source: Path | str) -> str:
        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400, "threshold": 0.5},
        }

        if self.language:
            transcribe_kwargs["language"] = self.language

        timer = TimerLine("[cyan]{spinner}[/cyan] transcribing [dim]{elapsed}[/dim]")
        try:
            with live_line(timer):
                segments, _ = self.model.transcribe(str(audio_source), **transcribe_kwargs)
                transcription_parts = [text for segment in segments if (text := segment.text.strip())]
        except Exception as e:
            raise RuntimeError(f"transcription failed: {e}") from e

        full_transcription = " ".join(transcription_parts)
        if not full_transcription:
            raise ValueError("no speech detected in audio")

        return full_transcription

    @classmethod
    def list_models(cls):
        for model in cls.VALID_MODELS:
            click.echo(model)
        console.print("\n[dim]environment variables:[/dim]")
        console.print("[dim]  HNS_WHISPER_MODEL=<model-name>   default model (base)[/dim]")
        console.print("[dim]  HNS_LANG=<language-code>         e.g. en, es, fr[/dim]")
        console.print("[dim]  HNS_CACHE_DIR=<dir>              last_recording.wav location[/dim]")


def copy_to_clipboard(text: str) -> bool:
    try:
        pyperclip.copy(text)
    except Exception as e:
        console.print(f"[yellow]![/yellow] clipboard copy failed: {escape(str(e))}")
        if sys.platform.startswith("linux"):
            console.print("  [dim]install xclip, xsel, or wl-clipboard for clipboard support[/dim]")
        return False
    return True


@click.command()
@click.version_option(package_name="hns", prog_name="hns")
@click.option(
    "--model",
    "-m",
    type=click.Choice(WhisperTranscriber.VALID_MODELS),
    help="Whisper model to use. Defaults to HNS_WHISPER_MODEL env var or 'base'",
)
@click.option("--language", help="Force language detection (e.g., en, es, fr). Can also use HNS_LANG env var")
@click.option("--last", is_flag=True, help="Transcribe the last recorded audio file")
@click.option("--no-copy", is_flag=True, help="Do not copy the transcription to the clipboard")
@click.option("--list-models", is_flag=True, help="List available Whisper models and exit")
def main(model: str | None, language: str | None, last: bool, no_copy: bool, list_models: bool):
    """Record audio from microphone, transcribe it, and copy to clipboard."""

    if list_models:
        WhisperTranscriber.list_models()
        return

    try:
        recorded_duration = None
        if last:
            audio_file_path = get_audio_file_path()
            if not audio_file_path.exists():
                console.print("[red]✗[/red] no previous recording found · run hns without --last to record one")
                sys.exit(1)
        else:
            recorder = AudioRecorder()
            audio_file_path = recorder.record()
            recorded_duration = recorder.duration

        transcriber = WhisperTranscriber(model_name=model, language=language)
        transcribe_start = time.time()
        transcription = transcriber.transcribe(audio_file_path)
        transcribe_seconds = time.time() - transcribe_start

        copied = copy_to_clipboard(transcription) if not no_copy else False

        summary = []
        if recorded_duration is not None:
            summary.append(f"recorded {format_duration(recorded_duration)}")
        summary.append(f"transcribed in {transcribe_seconds:.1f}s")
        if copied:
            summary.append("copied to clipboard")
        console.print(f"[green]✓[/green] [dim]{' · '.join(summary)}[/dim]")
        console.print()
        click.echo(transcription)

    except KeyboardInterrupt:
        console.print("[yellow]![/yellow] cancelled")
        sys.exit(130)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]✗[/red] {escape(str(e))}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] unexpected error: {escape(str(e))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
