# Quoridor

A from-scratch implementation of [Quoridor](https://quoridorstrats.wordpress.com/beginners-guide-rules-and-basics/),
built layer by layer: board state, then rendering, then rules engine, then a
pluggable player/renderer architecture, then a client-server split so a game
can be played across processes (or containers) instead of just in one.

## Architecture

```
quoridor/
  board.py        QuoridorBoard, BoardState — game state + (de)serialization
  pathfinding.py   is_wall_between, bfs_shortest_path — pure functions, no
                    project imports. The one place "how walls block movement"
                    is implemented; everything else calls into it instead of
                    keeping its own copy of the rule.
  rendering.py      Renderer (ABC) + CLIRenderer — turns a BoardState into a
                     human-facing view. render() is pure: no printing, no I/O.
  engine.py          Direction, WallOrientation, InvalidMoveError,
                       QuoridorEngine (the rules layer), and EngineLike — the
                       Protocol both QuoridorEngine and RemoteEngine satisfy.
  actions.py          MoveAction / WallAction / Action, apply_action()
  agents.py            Agent (ABC) + CLIAgent (human, via terminal input) +
                         BFSAgent (rule-based AI, shallow — always steps
                         along the shortest unobstructed path to its goal
                         row, never places walls)
  runner.py             GameRunner — alternates two Agents against one
                          EngineLike until there's a winner
  server.py              stdlib http.server game server — owns one
                           QuoridorEngine, exposes it over HTTP
  client.py               RemoteEngine — EngineLike implementation that
                            talks to a running server (urllib + json only)
  play_local.py            entrypoint: two Agents vs one local engine
  play_remote.py            entrypoint: one Agent vs a remote server
```

**The key idea:** `Agent`, `Renderer`, `GameRunner`, `play_local.py`, and
`play_remote.py` are all written against `EngineLike`, not against
`QuoridorEngine` directly. `QuoridorEngine` (in-process) and `RemoteEngine`
(talks to a server over HTTP) are two different implementations of that same
interface, so exactly the same `Agent`/`GameRunner` code plays a local game
or a networked one — `tests/test_integration.py` proves this by running the
same `GameRunner` two ways: once against a local engine, once against a real
running server.

## Rules implemented

- Turn-based pawn movement in the four cardinal directions, blocked by walls,
  board edges, and the opponent's pawn.
- Wall placement, validated against: remaining inventory, board bounds,
  overlap with an existing wall, crossing an existing wall, and — the one
  rule that goes beyond bare mechanics — a wall may never fully block either
  player's path to their goal row (checked with a BFS after tentatively
  placing the wall).
- Win detection when a pawn reaches the far row.

## Not implemented yet

- Diagonal moves / jumping over an adjacent opponent pawn.
- A real GUI (`CLIRenderer` is the only `Renderer` so far — the `Renderer`
  base class exists specifically so a GUI can be added as a second
  implementation without touching `Agent`/`GameRunner`/the engine).
- Support for 2 or 4 players.
- A stronger AI (`BFSAgent` is deliberately shallow — one BFS toward its own
  goal, no wall strategy, no lookahead, no opponent modeling). A future
  version estimating win probability per position (and pruning search with
  that estimate) is a natural next step.

## Requirements

- Python 3.13+ (containers pin `python:3.13-slim`; local dev works from 3.11+)
- No external runtime dependencies — the game, server, and client are all
  pure standard library (`http.server`, `urllib`, `json`). `make` sets up an
  isolated `.venv` for the dev tools only (pytest, mypy, ruff), since this
  system's Python is externally managed and won't allow global `pip install`.
- Docker + Docker Compose, if you want to run the containerized version.

## Usage

```bash
make install            # create .venv and install dev dependencies
make test                # run the pytest suite
make typecheck           # run mypy in strict mode
make lint                # run ruff
make check               # lint + typecheck + test — run this before committing
make run-local           # play a local game (human vs bfs by default)
make run-server          # start the HTTP game server on :8765
make run-client-human    # connect a human player to a running server
make run-client-bfs      # connect a BFS AI player to a running server
make docker-build        # build the server/client images
make docker-up           # start server + AI client in Docker
make docker-down         # stop and remove the Docker stack
make clean               # remove the venv and all caches
```

Run `make help` to see this list from the terminal.

### Playing locally, without a server

```bash
python -m quoridor.play_local --p1 human --p2 bfs --size 9
```

`--p1`/`--p2` are each `human` or `bfs`. A human turn shows the board and
prompts for a command: `move <up|down|left|right>` or `wall <h|v> <row> <col>`.

### Playing over the network, without Docker

```bash
python -m quoridor.server --port 8765 --size 9        # terminal 1
python -m quoridor.play_remote --url http://localhost:8765 --player 1 --agent human   # terminal 2
python -m quoridor.play_remote --url http://localhost:8765 --player 2 --agent bfs     # terminal 3
```

Each `play_remote` process only acts for its own player number — it polls
the server and waits until it's that player's turn.

### Playing in Docker

```bash
docker compose build
docker compose up -d server ai-client   # start the server + an AI opponent
docker compose run --rm cli-client      # attach as the human (player 1)
```

`cli-client` isn't started by plain `up` on purpose — Compose doesn't attach
a TTY well to one service among several, so the interactive human is run
separately with `run --rm` instead. `RemoteEngine` retries against `/health`
on startup, so client containers don't need to race the server's boot time.

## Type checking

Game code (everything under `quoridor/`) is fully type-hinted and checked
with `mypy --strict`, including a `Protocol`-based static check
(`quoridor/client.py`, under `if TYPE_CHECKING:`) that both `QuoridorEngine`
and `RemoteEngine` genuinely satisfy `EngineLike` — not just "probably do".

This strictness is a direct response to a real bug hit early on:
`QuoridorBoard.from_dict` was reassigning `h_walls`/`v_walls` (declared as
`set[tuple[int, int]]`) straight from JSON-shaped lists, so a restored board
silently carried lists where the rest of the code expected sets. The fix was
twofold — correct the bug, and make it structurally impossible to
reintroduce quietly: `to_dict`/`from_dict` go through an explicit
`BoardState` `TypedDict` instead of an untyped `dict`, and `from_dict`
assigns each field explicitly (no generic `setattr` loop) so mypy can catch
a type mismatch on that boundary. `make check` runs `mypy` on every change.

Test files under `tests/` are exempted from mypy's "must annotate every def"
rule (conventional for pytest-style tests) but still get everything else
strict mode checks.

## Tests

`tests/` (pytest, run via `make test`) covers every layer:

- `test_board.py`, `test_pathfinding.py` — state serialization, movement/BFS
  rules in isolation.
- `test_engine.py` — the same wall-blocking rules exercised through
  `QuoridorEngine`, proving it wires the shared `pathfinding` functions in
  correctly (deliberately overlapping with `test_pathfinding.py`, not
  redundant — one proves the rule, the other proves the wiring).
- `test_rendering.py`, `test_agents.py`, `test_runner.py` — the renderer,
  both `Agent` implementations (including `CLIAgent`'s input re-prompt loop
  via a monkeypatched `input()`), and `GameRunner`.
- `test_server.py` — every HTTP route against a real `ThreadingHTTPServer`
  spun up on an ephemeral port per test.
- `test_integration.py` — a full `BFSAgent` vs `BFSAgent` game driven
  through `GameRunner` + `RemoteEngine` against a real running server, plus
  a direct check that a `409` response becomes a local `InvalidMoveError`.
