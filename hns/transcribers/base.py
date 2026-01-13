from typing import Protocol, Union, Optional
from pathlib import Path


class TranscriberBackend(Protocol):
    def transcribe(self, audio_source: Union[Path, str], show_progress: bool = True) -> tuple[str, Optional[float]]:
        """Transcribe audio file, return (text, elapsed_seconds)."""
        ...

    @classmethod
    def list_models(cls) -> None:
        """Print available models to console."""
        ...

    @classmethod
    def get_valid_models(cls) -> list[str]:
        """Return list of valid model names for this backend."""
        ...
