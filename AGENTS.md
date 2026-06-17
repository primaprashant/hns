# AGENTS.md

## Project

`hns` is a small, privacy-focused speech-to-text CLI. It records microphone audio, transcribes it locally with `faster-whisper`, prints the transcription to stdout, and copies it to the clipboard.

Keep the tool simple, local-first, and composable with other CLI tools. Prefer small changes that preserve the current UX and Unix-style separation: user-facing progress/status goes to stderr, the final transcription goes to stdout.

## Where To Look

- `README.md`: product overview, user-facing behavior, installation, use cases, and docs links.
- `hns/cli.py`: complete implementation of recording, transcription, clipboard copy, and Click CLI options.
- `pyproject.toml`: package metadata, dependencies, console script, Python/Ruff settings.
- `Makefile`: common development, lint, version, build, and publish commands.
- `.github/workflows/`: CI lint/format check and PyPI publish workflow.

## Tech Stack

- Python `>=3.10`; `.python-version` currently pins `3.10`.
- Packaging and environments use `uv` with `uv.lock` checked in.
- CLI uses `click`; terminal output uses `rich`.
- Audio capture uses `sounddevice`, `wave`, and `numpy`.
- Transcription uses `faster-whisper` on CPU with `int8` compute.
- Clipboard integration uses `pyperclip`.

## Development Commands

- Install/sync dev environment: `uv sync --dev`
- Add runtime dependencies: `uv add <package>`
- Add development dependencies: `uv add --dev <package>`
- Run the CLI locally: `uv run --project . python -m hns.cli`
- Run with the Makefile default model: `make run`
- Show CLI help: `make test-help`
- List Whisper models: `make list-models`
- Optional transcription fixture check: `make test-transcription`
- Format and lint: `make format-and-lint`
- Build package: `uv build`
- Bump version and lockfile: `make bump-patch` or `make bump-minor`

There is no full automated test suite. For behavior changes, run `make test-help`, `make list-models`, and `make format-and-lint`. For transcription pipeline changes, also run `make test-transcription`; it uses `tests/fixtures/last_recording.wav` through `hns --last` and does not require microphone hardware. Manually exercise recording flows when audio hardware is available.

## Runtime Configuration

- `HNS_WHISPER_MODEL`: default Whisper model when no explicit model is passed internally; invalid values fall back to `base`.
- `HNS_LANG`: optional language code passed to Whisper, e.g. `en`, `es`, `fr`.
- `HNS_CACHE_DIR`: optional cache directory override for `last_recording.wav`, mainly useful for isolated fixture tests.
- `--language`: CLI override for transcription language.
- `--last`: retranscribes the cached last recording instead of recording new audio.

Cached audio is written to the platform cache directory as `last_recording.wav`.

## Code Organization

- `format_duration`: shared timer formatting.
- `AudioRecorder`: validates input device, records microphone audio, writes WAV data, manages cache path.
- `WhisperTranscriber`: validates model names, loads `faster-whisper`, transcribes audio, lists models.
- `copy_to_clipboard`: copies final text and reports clipboard status.
- `main`: Click command wiring, option handling, error handling, stdout/stderr behavior.
- `tests/check_transcription_fixture.py`: optional non-microphone transcription smoke check using committed fixtures.

## Fixture Testing

- Use `make test-transcription` when changing transcription behavior, model handling, VAD settings, or `--last` behavior.
- The fixture test is configurable, for example: `uv run --no-sync --project . python tests/check_transcription_fixture.py --model tiny.en --language en --min-similarity 0.50`.
- The check compares normalized word similarity, not exact text; update `tests/fixtures/last_recording.txt` only when the fixture audio intentionally changes.

## Conventions

- Keep implementation compact; avoid adding modules unless complexity clearly justifies it.
- Preserve stdout for machine-readable transcription output only.
- Preserve stderr for progress, errors, warnings, and status messages through the `rich` stderr console.
- Prefer explicit, user-friendly CLI errors over tracebacks.
- Keep dependencies minimal and justified; this is a simple CLI distributed on PyPI.
- Manage dependencies with `uv add` or `uv add --dev`; do not edit `pyproject.toml` directly for dependency changes.
- Follow Ruff settings in `pyproject.toml` (`line-length = 120`, target `py310`).

## Change Carefully

- Recording/transcription paths are interactive and hardware-dependent; avoid breaking non-audio commands like `--help` and `--list-models`.
- First transcription may download a Whisper model; do not add network requirements beyond model download behavior.
- Clipboard failures should not prevent printing transcription to stdout.
- Cross-platform cache paths in `AudioRecorder._get_audio_file_path` affect `--last` behavior.
- Published package behavior comes from the `hns = "hns.cli:main"` console script in `pyproject.toml`.
