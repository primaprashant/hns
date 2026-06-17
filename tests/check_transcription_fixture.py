import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = PROJECT_ROOT / "tests" / "fixtures" / "last_recording.wav"
DEFAULT_EXPECTED = PROJECT_ROOT / "tests" / "fixtures" / "last_recording.txt"


def normalize(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return normalized.split()


def word_similarity(expected: str, actual: str) -> float:
    expected_words = normalize(expected)
    actual_words = normalize(actual)

    if not expected_words and not actual_words:
        return 1.0
    if not expected_words or not actual_words:
        return 0.0

    previous = list(range(len(actual_words) + 1))
    for i, expected_word in enumerate(expected_words, start=1):
        current = [i]
        for j, actual_word in enumerate(actual_words, start=1):
            substitution_cost = 0 if expected_word == actual_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current

    distance = previous[-1]
    return max(0.0, 1.0 - (distance / max(len(expected_words), len(actual_words))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hns --last against a committed audio fixture.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Audio fixture to copy as last_recording.wav")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED, help="Ground-truth transcription text")
    parser.add_argument("--model", default=os.environ.get("HNS_TEST_MODEL"), help="Whisper model for this check")
    parser.add_argument("--language", default=os.environ.get("HNS_TEST_LANGUAGE"), help="Language code passed to hns")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=float(os.environ.get("HNS_TEST_MIN_SIMILARITY", "0.70")),
        help="Minimum normalized word similarity required to pass",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = args.expected.read_text(encoding="utf-8").strip()

    with tempfile.TemporaryDirectory(prefix="hns-fixture-") as cache_dir:
        cache_path = Path(cache_dir)
        shutil.copy2(args.audio, cache_path / "last_recording.wav")

        env = os.environ.copy()
        env["HNS_CACHE_DIR"] = str(cache_path)
        if args.model:
            env["HNS_WHISPER_MODEL"] = args.model

        command = [sys.executable, "-m", "hns.cli", "--last"]
        if args.language:
            command.extend(["--language", args.language])

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    actual = completed.stdout.strip()
    similarity = word_similarity(expected, actual)
    model = args.model or env.get("HNS_WHISPER_MODEL", "hns default")

    print(f"Model: {model}")
    print(f"Similarity: {similarity:.2f} (minimum {args.min_similarity:.2f})")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")

    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        return completed.returncode

    if similarity < args.min_similarity:
        print(completed.stderr, file=sys.stderr)
        print("Transcription fixture check failed: similarity is below threshold.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
