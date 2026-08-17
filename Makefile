VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

URL ?= http://localhost:8765

.PHONY: help install test typecheck lint check \
	run-local run-server run-client-human run-client-bfs \
	docker-build docker-up docker-down clean

help:
	@echo "Available targets:"
	@echo "  make install           Create venv and install dependencies"
	@echo "  make test              Run the test suite"
	@echo "  make typecheck         Run mypy (strict) over the source"
	@echo "  make lint              Run ruff"
	@echo "  make check             Run lint + typecheck + test (use before committing)"
	@echo "  make run-local         Play a local game (human vs bfs by default)"
	@echo "  make run-server        Start the Quoridor HTTP game server"
	@echo "  make run-client-human  Connect a human player to a running server"
	@echo "                         (defaults to PLAYER=1; override with PLAYER=2, URL=...)"
	@echo "  make run-client-bfs    Connect a BFS AI player to a running server"
	@echo "                         (defaults to PLAYER=2; override with PLAYER=1, URL=...)"
	@echo "  make docker-build      Build the server/client Docker images"
	@echo "  make docker-up         Start server + AI client in Docker; attach a human with"
	@echo "                         'docker compose run --rm cli-client'"
	@echo "  make docker-down       Stop and remove the Docker Compose stack"
	@echo "  make clean             Remove venv and caches"

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

run-local: install
	$(PYTHON) -m quoridor.play_local --p1 human --p2 bfs

run-server: install
	$(PYTHON) -m quoridor.server

run-client-human: install
	$(PYTHON) -m quoridor.play_remote --url $(URL) --player $(if $(PLAYER),$(PLAYER),1) --agent human

run-client-bfs: install
	$(PYTHON) -m quoridor.play_remote --url $(URL) --player $(if $(PLAYER),$(PLAYER),2) --agent bfs

docker-build:
	docker compose build

docker-up:
	docker compose up -d server ai-client

docker-down:
	docker compose down

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache __pycache__ quoridor/__pycache__ tests/__pycache__
