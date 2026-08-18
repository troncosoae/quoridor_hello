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
  runner.py             GameRunner — cycles N Agents (2 or 4) against one
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

## Technical specifications

### Where a game actually starts

- **Local:** `play_local.py:29` builds `QuoridorEngine(QuoridorBoard(...))`,
  then `play_local.py:35-36` hands it to `GameRunner(engine, agents).run()`,
  which loops until there's a winner.
- **Remote:** the server creates the one shared `QuoridorEngine` at
  `server.py:152` inside `main()`, before `serve_forever()`. Each client only
  starts acting once it calls `remote.claim_player(...)`
  (`play_remote.py:32`) and enters its polling loop (`play_remote.py:42`).

### `GameServer` mechanics

`GameServer` (`server.py:20`) subclasses `ThreadingHTTPServer`, so every
incoming request runs on its own thread instead of being handled one at a
time — required here since multiple players poll and act concurrently. All
threads share the single `QuoridorEngine` instance stored on the server, so
`GameServer.lock` (`server.py:27`, a plain `threading.Lock`) guards every
read/write of engine state: each handler wraps its logic in
`with self._server.lock:` before touching `engine`, which prevents two
requests (e.g. two moves) from racing each other.

### Sequence: client connects and claims a seat

```mermaid
sequenceDiagram
    participant S as Server
    participant A as Client A
    participant B as Client B

    Note over S: main() creates engine, serve_forever()
    A->>S: GET /health
    S-->>A: 200 ok
    A->>S: POST /claim (no player = auto-assign)
    Note over S: lock; assign next open seat
    S-->>A: 200 {player: 1}

    B->>S: GET /health
    S-->>B: 200 ok
    B->>S: POST /claim (no player = auto-assign)
    Note over S: lock; assign next open seat
    S-->>B: 200 {player: 2}
```

### Sequence: one action, through to the next player acting

Nothing is pushed to clients — each `play_remote.py` process polls
`/state` every `POLL_INTERVAL` second (`play_remote.py:42-45`) and only acts
once it sees `current_player` equal to its own seat.

```mermaid
sequenceDiagram
    participant A as Client A (current player)
    participant S as Server
    participant B as Client B (waiting)

    A->>S: GET /state
    S-->>A: 200 {current_player: A}
    Note over A: agent.choose_action()
    A->>S: POST /move
    Note over S: lock; engine.move(); current_player -> B; unlock
    S-->>A: 200 {current_player: B}
    Note over A: current_player != A, sleep(POLL_INTERVAL)

    B->>S: GET /state
    S-->>B: 200 {current_player: B}
    Note over B: sees it's their turn
    B->>S: POST /move (or /wall)
    Note over S: lock; engine mutates...; unlock
```

## Rules implemented

- 2-player and 4-player games. In 4-player, seats 1/2 start at the top/bottom
  edge (goal: the opposite row) and seats 3/4 start at the left/right edge
  (goal: the opposite column) — same board, same rules, just more seats.
  Wall counts scale with player count: half the 2-player allowance per seat
  (e.g. 9×9 is 10 walls each for 2 players, 5 each for 4 — matching real
  Quoridor's documented 4-player rule).
- Turn-based pawn movement in the four cardinal directions, blocked by walls,
  board edges, and any other player's pawn.
- Wall placement, validated against: remaining inventory, board bounds,
  overlap with an existing wall, crossing an existing wall, and — the one
  rule that goes beyond bare mechanics — a wall may never fully block *any*
  player's path to their goal (checked via a full-board connected-components
  scan after tentatively placing the wall).
- Win detection when a pawn reaches its goal edge.
- Server-side seat assignment: connecting without specifying a player claims
  the next open seat in connection order; a specific seat can still be
  requested explicitly, and a taken (or, once full, nonexistent) seat is
  rejected either way.

## Not implemented yet

- Diagonal moves / jumping over an adjacent opponent pawn.
- A real GUI (`CLIRenderer` is the only `Renderer` so far — the `Renderer`
  base class exists specifically so a GUI can be added as a second
  implementation without touching `Agent`/`GameRunner`/the engine).
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
python -m quoridor.play_local --players 4 --agents human bfs bfs bfs --size 9
```

`--players` is `2` or `4` (default 2); `--agents` takes one `human`/`bfs`
token per seat (default `human bfs`). A human turn shows the board and
prompts for a command: `move <up|down|left|right>` or `wall <h|v> <row> <col>`.

### Playing over the network, without Docker

```bash
python -m quoridor.server --port 8765 --size 9 --players 4   # terminal 1
python -m quoridor.play_remote --url http://localhost:8765 --agent human   # terminal 2
python -m quoridor.play_remote --url http://localhost:8765 --agent bfs     # terminal 3
python -m quoridor.play_remote --url http://localhost:8765 --agent bfs     # terminal 4
python -m quoridor.play_remote --url http://localhost:8765 --agent bfs     # terminal 5
```

Each `play_remote` process claims one seat and only acts for it — polling
the server and waiting until it's that seat's turn. Omit `--player` (as
above) to auto-assign the next open seat in connection order, or pass
`--player N` to claim a specific one explicitly; either way the process
prints which seat it ended up with. A game needs every configured seat
filled by an active client to make progress — the server has no game loop of
its own, it's purely reactive, so an empty seat just means the game stalls
on that player's turn until someone connects for it.

### Playing in Docker

```bash
docker compose build
docker compose up -d server ai-client-1 ai-client-2 ai-client-3   # server + 3 AI seats
docker compose run --rm cli-client                                # attach as the human
```

The default `docker-compose.yml` is a 4-player game: the server runs with
`--players 4`, and none of the four client services pass `--player` — seats
auto-assign, so whichever container connects first becomes seat 1, and so
on (the `ai-client-N` names are just Compose service identifiers, not fixed
game seats). `cli-client` isn't started by plain `up` on purpose — Compose
doesn't attach a TTY well to one service among several, so the interactive
human is run separately with `run --rm` instead. `RemoteEngine` retries
against `/health` on startup, so client containers don't need to race the
server's boot time.

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
- `test_integration.py` — full `BFSAgent`-only games (both 2-player and
  4-player) driven through `GameRunner` + `RemoteEngine` against a real
  running server, plus direct checks that a `409` response becomes a local
  `InvalidMoveError`/`SeatTakenError` as appropriate.
