from .whisper import WhisperTranscriber


def create_transcriber(model_name: str = None, language: str = None):
    return WhisperTranscriber(model_name=model_name, language=language)


def list_all_models():
    WhisperTranscriber.list_models()
