# hns - A simple, privacy-focused speech-to-text CLI tool

A simple, privacy-focused speech-to-text CLI tool that records your voice, transcribes it locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and copies the transcription to your clipboard. Perfect for interacting with Claude Code, VS Code, Slack, or any application where native dictation falls short.

Click the image below to see the demo ⬇️ (warning: demo has sound)

[![Watch Demo](https://cdn.anandprashant.com/hns/thumbnail-1.0.1.png)](https://cdn.anandprashant.com/hns/demo-1.0.1.mp4)

## Highlights

- **100% Local & Private**: Audio is processed entirely on your local machine. No data leaves your device
- **Works Offline**: After the initial model download, no internet connection is required
- **Instant Clipboard**: Transcribed text is automatically copied to your clipboard for immediate pasting
- **Multi-Language Support**: Transcribe in any language supported by Whisper
- **Configurable**: Choose models and languages via environment variables
- **Focused** - Does one thing well: speech → clipboard
- **Open Source** - MIT licensed, fully transparent

## Use Cases

- **Claude Code & AI Assistants**: Perfect for Claude Code or any AI interface without native dictation. Run `hns`, speak your prompt, then paste into Claude Code.
- **Brain Dump → Structured Output**: Ramble your scattered thoughts, then paste into an LLM to organize:
  ```sh
  hns  # "So I'm thinking about the refactor... we need to handle auth, but also consider caching..."
  # Paste to LLM: "Create a structured plan from these thoughts:"
  ```
- **Communication**: Compose Slack messages, emails, or chat responses hands-free.
- **Note-Taking**: Quickly capture thoughts and ideas without switching from the keyboard.
- **Accessibility**: Helpful for users who find typing difficult or painful.

## Installation

Install via [uv](https://github.com/astral-sh/uv) (recommended):
```sh
uv tool install hns
```
or `pipx`:
```sh
pipx install hns
```
or `pip`:
```sh
pip install --user hns
```

The first time you run `hns`, it will download the default Whisper model (`base`). This requires an internet connection and may take a few moments. Subsequent runs can be fully offline.

## Usage

### Basic Transcription

1.  Run the command in your terminal:
    ```sh
    hns
    ```
2.  The tool will display `🎤 Recording...`. Speak into your microphone.
3.  Press `Enter` when you have finished.
4.  The transcribed text is automatically copied to your clipboard and printed to the console.

### Configuration

#### Listing Models

To see all available transcription models:
```sh
hns --list-models
```

#### Setting the Model

Select a model by setting the `HNS_WHISPER_MODEL` environment variable. The default is `base`. For higher accuracy, use a larger model like `medium` or `large-v3`.

```sh
# Use the 'small' model for the current session
export HNS_WHISPER_MODEL="small"
hns
```

To make the change permanent, add `export HNS_WHISPER_MODEL="<model_name>"` to your shell profile (`.zshrc`, `.bash_profile`, etc.).

#### Setting the Language

By default, Whisper auto-detects the language. To force a specific language, set the `HNS_LANG` environment variable.

```sh
# Use an environment variable for Japanese
export HNS_LANG="ja"
hns
```

## License

This project is licensed under the [MIT License](./LICENSE).
