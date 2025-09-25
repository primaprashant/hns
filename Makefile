.PHONY: run test-help list-models test-tiny

run:
	@HNS_WHISPER_MODEL=small uv run --no-sync --project . python -m hns.cli

test-help:
	@uv run --no-sync --project . python -m hns.cli --help

list-models:
	@uv run --no-sync --project . python -m hns.cli --list-models

format-and-lint:
	uv run --no-sync --project . ruff check --select I --fix
	uv run --no-sync --project . ruff format
	uv run --no-sync --project . ruff check --fix

bump-patch:
	uv version --bump patch
	uv sync
	@echo "Patch version bumped and recreated lock file."

bump-minor:
	uv version --bump minor
	uv sync
	@echo "Minor version bumped and recreated lock file."

publish-test:
	uv build
	@if [ -z "$(PYPI_TEST_TOKEN)" ]; then echo "Error: TEST_PYPI_TOKEN is not set"; exit 1; fi
	uv publish --publish-url https://test.pypi.org/legacy/ --token ${PYPI_TEST_TOKEN}

publish:
	uv build
	@if [ -z "$(PYPI_TOKEN)" ]; then echo "Error: PYPI_TOKEN is not set"; exit 1; fi
	uv publish --token ${PYPI_TOKEN}
