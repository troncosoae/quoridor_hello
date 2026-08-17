VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help install test typecheck lint check run-board run-engine clean

help:
	@echo "Available targets:"
	@echo "  make install     Create venv and install dependencies"
	@echo "  make test        Run the test suite"
	@echo "  make typecheck   Run mypy (strict) over the source"
	@echo "  make lint        Run ruff"
	@echo "  make check       Run lint + typecheck + test (use before committing)"
	@echo "  make run-board   Render a sample board (board.py demo)"
	@echo "  make run-engine  Run the engine demo (moves + wall placement)"
	@echo "  make clean       Remove venv and caches"

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	touch $(VENV)/bin/activate

install: $(VENV)/bin/activate

test: install
	$(PYTHON) -m pytest

typecheck: install
	$(PYTHON) -m mypy .

lint: install
	$(PYTHON) -m ruff check .

check: lint typecheck test

run-board: install
	$(PYTHON) board.py

run-engine: install
	$(PYTHON) engine.py

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache __pycache__ tests/__pycache__
