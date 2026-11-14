.PHONY: install fixture test smoke lint format typecheck check

install:
	python -m pip install -e ".[dev]"

fixture:
	python scripts/create_synthetic_fixture.py

test:
	python -m pytest -m "not slow"

smoke:
	python -m pytest tests/integration/test_smoke_pipeline.py -m slow -v

lint:
	python -m ruff check .

format:
	python -m ruff format .

typecheck:
	python -m mypy

check: lint typecheck test smoke

