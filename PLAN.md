# Implementation Plan: Multi-Backend ASR Support (Whisper + Parakeet)

This plan adds NVIDIA Parakeet model support while keeping the codebase maintainable. Each phase is independent and results in a working system.

---

## Phase 1: Create Transcriber Abstraction Layer

**Goal:** Extract transcriber interface without changing any behavior.

### Changes

1. **Create `hns/transcribers/` package:**
   ```
   hns/transcribers/
   ├── __init__.py      # Exports + factory function
   ├── base.py          # TranscriberBackend Protocol
   └── whisper.py       # Move WhisperTranscriber here
   ```

2. **`base.py`** - Define the interface:
   ```python
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
   ```

3. **`whisper.py`** - Move `WhisperTranscriber` class from `cli.py` (lines 139-275) with no changes.

4. **`__init__.py`** - Simple re-export:
   ```python
   from .whisper import WhisperTranscriber

   def create_transcriber(model_name: str = None, language: str = None):
       return WhisperTranscriber(model_name=model_name, language=language)

   def list_all_models():
       WhisperTranscriber.list_models()
   ```

5. **Update `cli.py`:**
   - Replace `from .transcribers import create_transcriber, list_all_models`
   - Change `WhisperTranscriber(language=language)` → `create_transcriber(language=language)`
   - Change `WhisperTranscriber.list_models()` → `list_all_models()`

### Verification
- `hns` works exactly as before
- `hns --list-models` shows Whisper models
- All existing env vars (`HNS_WHISPER_MODEL`, `HNS_LANG`) work

---

## Phase 2: Unify Model Selection with `--model` Flag

**Goal:** Add explicit `--model` CLI option (currently model is only set via env var).

### Changes

1. **Add `--model` option to CLI:**
   ```python
   @click.option("--model", "-m", help="Model to use for transcription (e.g., base, large-v3)")
   ```

2. **Update `create_transcriber()` to accept model_name:**
   ```python
   transcriber = create_transcriber(model_name=model, language=language)
   ```

3. **Rename env var** (with backward compat):
   - Primary: `HNS_MODEL`
   - Fallback: `HNS_WHISPER_MODEL` (for backward compatibility)

   In `whisper.py`:
   ```python
   model = model_name or os.environ.get("HNS_MODEL") or os.environ.get("HNS_WHISPER_MODEL", "base")
   ```

### Verification
- `hns --model large-v3` uses large-v3
- `HNS_MODEL=small hns` uses small
- `HNS_WHISPER_MODEL=medium hns` still works (backward compat)

---

## Phase 3: Add Parakeet Transcriber

**Goal:** Implement Parakeet backend, auto-detect backend from model name.

### Parakeet Model Names
```python
PARAKEET_MODELS = [
    "parakeet-ctc-0.6b",
    "parakeet-ctc-1.1b",
    "parakeet-rnnt-0.6b",
    "parakeet-rnnt-1.1b",
    "parakeet-tdt-1.1b",
    "parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v3",
    "parakeet-tdt_ctc-0.6b",
    "parakeet-tdt_ctc-1.1b",
    "parakeet-tdt_ctc-110m",
]
```

### Changes

1. **Create `hns/transcribers/parakeet.py`:**
   ```python
   import os
   import time
   from pathlib import Path
   from typing import Optional, Union

   from rich.console import Console

   console = Console(stderr=True)


   class ParakeetTranscriber:
       VALID_MODELS = [
           "parakeet-ctc-0.6b",
           "parakeet-ctc-1.1b",
           "parakeet-rnnt-0.6b",
           "parakeet-rnnt-1.1b",
           "parakeet-tdt-1.1b",
           "parakeet-tdt-0.6b-v2",
           "parakeet-tdt-0.6b-v3",
           "parakeet-tdt_ctc-0.6b",
           "parakeet-tdt_ctc-1.1b",
           "parakeet-tdt_ctc-110m",
       ]

       def __init__(self, model_name: str, language: Optional[str] = None):
           self.model_name = self._get_model_name(model_name)
           self.language = language  # Note: Parakeet v2 is English-only, v3 is multilingual
           self.model = None  # Lazy load

       def _get_model_name(self, model_name: str) -> str:
           if model_name not in self.VALID_MODELS:
               console.print(f"⚠️ [bold yellow]Invalid Parakeet model '{model_name}'[/bold yellow]")
               console.print(f"    [dim]Available: {', '.join(self.VALID_MODELS)}[/dim]")
               raise ValueError(f"Invalid Parakeet model: {model_name}")
           return model_name

       def _load_model(self):
           if self.model is not None:
               return self.model

           try:
               import nemo.collections.asr as nemo_asr
           except ImportError:
               raise RuntimeError(
                   "Parakeet models require nemo_toolkit. "
                   "Install with: pip install 'hns[parakeet]' or pip install nemo_toolkit[asr]"
               )

           try:
               self.model = nemo_asr.models.ASRModel.from_pretrained(
                   model_name=f"nvidia/{self.model_name}"
               )
               return self.model
           except Exception as e:
               raise RuntimeError(f"Failed to load Parakeet model: {e}")

       def transcribe(self, audio_source: Union[Path, str], show_progress: bool = True) -> tuple[str, Optional[float]]:
           model = self._load_model()
           start_time = time.time()

           # Similar threading pattern as WhisperTranscriber for progress display
           if show_progress:
               import queue
               import threading
               from ..cli import format_duration

               progress_queue = queue.Queue()
               transcription_complete = threading.Event()

               def transcribe_worker():
                   try:
                       output = model.transcribe([str(audio_source)])
                       text = output[0].text if hasattr(output[0], 'text') else output[0]
                       progress_queue.put(("result", text))
                   except Exception as e:
                       progress_queue.put(("error", e))
                   finally:
                       transcription_complete.set()

               worker_thread = threading.Thread(target=transcribe_worker)
               worker_thread.daemon = True
               worker_thread.start()

               while not transcription_complete.is_set():
                   elapsed = time.time() - start_time
                   time_str = format_duration(elapsed)
                   console.print(f"🔄 [bold blue]Transcribing ... {time_str}[/bold blue]", end="\r")
                   time.sleep(1)

               console.print("")
               result_type, result_data = progress_queue.get()
               if result_type == "error":
                   raise result_data
               transcription = result_data
           else:
               output = model.transcribe([str(audio_source)])
               transcription = output[0].text if hasattr(output[0], 'text') else output[0]

           if not transcription or not transcription.strip():
               raise ValueError("No speech detected in audio")

           elapsed_total = time.time() - start_time
           return transcription.strip(), elapsed_total if show_progress else None

       @classmethod
       def list_models(cls):
           console.print("ℹ️ [bold cyan]Available Parakeet models:[/bold cyan]")
           for model in cls.VALID_MODELS:
               console.print(f"  • [dim]{model}[/dim]")

       @classmethod
       def get_valid_models(cls) -> list[str]:
           return cls.VALID_MODELS.copy()
   ```

2. **Update `hns/transcribers/__init__.py`:**
   ```python
   from .whisper import WhisperTranscriber

   # Lazy import for Parakeet to avoid loading nemo unnecessarily
   def _get_parakeet_transcriber():
       from .parakeet import ParakeetTranscriber
       return ParakeetTranscriber


   def is_parakeet_model(model_name: str) -> bool:
       return model_name is not None and model_name.startswith("parakeet")


   def create_transcriber(model_name: str = None, language: str = None):
       if is_parakeet_model(model_name):
           ParakeetTranscriber = _get_parakeet_transcriber()
           return ParakeetTranscriber(model_name=model_name, language=language)
       else:
           return WhisperTranscriber(model_name=model_name, language=language)


   def list_all_models():
       WhisperTranscriber.list_models()
       print()  # Blank line between sections
       try:
           ParakeetTranscriber = _get_parakeet_transcriber()
           ParakeetTranscriber.list_models()
       except ImportError:
           from rich.console import Console
           console = Console(stderr=True)
           console.print("ℹ️ [dim]Parakeet models available with: pip install 'hns[parakeet]'[/dim]")


   def get_all_valid_models() -> list[str]:
       models = WhisperTranscriber.get_valid_models()
       try:
           ParakeetTranscriber = _get_parakeet_transcriber()
           models.extend(ParakeetTranscriber.get_valid_models())
       except ImportError:
           pass
       return models
   ```

3. **Update `pyproject.toml` for optional dependencies:**
   ```toml
   [project.optional-dependencies]
   parakeet = ["nemo_toolkit[asr]>=2.0.0"]
   all = ["nemo_toolkit[asr]>=2.0.0"]
   ```

### Verification
- `hns --model base` → Uses Whisper
- `hns --model parakeet-tdt-0.6b-v2` → Uses Parakeet
- `hns --list-models` → Shows both Whisper and Parakeet models
- `pip install hns` → Only installs faster-whisper (no NeMo)
- `pip install 'hns[parakeet]'` → Also installs NeMo

---

## Phase 4: Polish and Documentation

**Goal:** Improve error messages, update docs, add `get_valid_models()` to Whisper.

### Changes

1. **Add `get_valid_models()` to `WhisperTranscriber`:**
   ```python
   @classmethod
   def get_valid_models(cls) -> list[str]:
       return cls.VALID_MODELS.copy()
   ```

2. **Improve error messages for missing dependencies:**
   ```python
   # In create_transcriber, if Parakeet model requested but nemo not installed:
   raise RuntimeError(
       f"Model '{model_name}' requires nemo_toolkit.\n"
       "Install with: pip install 'hns[parakeet]'"
   )
   ```

3. **Update README.md:**
   - Document `--model` flag
   - Document `HNS_MODEL` env var
   - Add Parakeet installation instructions
   - List available Parakeet models
   - Note GPU requirements for Parakeet

4. **Update `--list-models` output format:**
   ```
   Whisper models:
     • tiny, tiny.en, base, base.en, small, small.en, ...

   Parakeet models (requires: pip install 'hns[parakeet]'):
     • parakeet-tdt-0.6b-v2, parakeet-tdt-0.6b-v3, ...

   Environment variables:
     HNS_MODEL=<model-name>
     HNS_LANG=<language-code>
   ```

### Verification
- Clear error when using Parakeet without NeMo installed
- README documents all options
- `--list-models` is informative

---

## Summary

| Phase | Scope | Breaking Changes | Can Ship Independently |
|-------|-------|------------------|------------------------|
| 1 | Refactor into transcribers package | None | Yes |
| 2 | Add `--model` CLI option | None | Yes |
| 3 | Add Parakeet backend | None | Yes |
| 4 | Polish, docs, error messages | None | Yes |

### Final File Structure
```
hns/
├── __init__.py
├── cli.py                 # AudioRecorder + CLI orchestration
└── transcribers/
    ├── __init__.py        # Factory + list_all_models()
    ├── base.py            # TranscriberBackend Protocol
    ├── whisper.py         # WhisperTranscriber
    └── parakeet.py        # ParakeetTranscriber
```

### Key Design Decisions

1. **Model name determines backend** - No separate `--backend` flag needed
2. **Lazy imports** - NeMo only loaded when Parakeet model requested
3. **Optional dependencies** - Base install only includes faster-whisper
4. **Backward compatible** - `HNS_WHISPER_MODEL` still works
5. **Same interface** - Both backends return `(text, elapsed_time)` tuple

---

## Sources

- [Parakeet TDT 0.6B V2 on HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA Parakeet Collection](https://huggingface.co/collections/nvidia/parakeet)
- [Parakeet TDT 0.6B V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NVIDIA NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
