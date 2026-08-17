# Quoridor

A from-scratch implementation of [Quoridor](https://quoridorstrats.wordpress.com/beginners-guide-rules-and-basics/),
built layer by layer: board state, then rendering, then rules engine, with a full UI
still to come.

## What's here so far

- **`board.py`** — `QuoridorBoard`, the game state: board size (5/7/9, with the
  matching official wall count), pawn positions, placed walls, and each player's
  remaining wall inventory. Serializes to/from a plain dict (`BoardState`) for save/load
  or transport. Also has `CLIRenderer`, which prints the board as ASCII art — cells,
  pawns (`1`/`2`), and walls (`---` / `|`) — to the terminal.
- **`engine.py`** — `QuoridorEngine`, the rules layer on top of `QuoridorBoard`:
  - Turn-based pawn movement in the four cardinal directions, blocked by walls,
    board edges, and the opponent's pawn.
  - Wall placement, validated against: remaining inventory, board bounds, overlap
    with an existing wall, crossing an existing wall, and — the one rule that goes
    beyond bare mechanics — a wall may never fully block either player's path to
    their goal row (checked with a BFS after tentatively placing the wall).
  - Win detection when a pawn reaches the far row.
- **`brain_v1.py`** — stub for a future AI opponent (currently empty scaffolding).
- **`tests/`** — pytest suite covering `board.py` and `engine.py` (26 tests):
  construction, serialization round-trips, movement legality, wall legality, and
  win conditions.

## Not implemented yet

- Diagonal moves / jumping over an adjacent opponent pawn.
- A real UI (currently just the CLI board renderer — no input loop, no full game yet).
- The AI opponent (`brain_v1.py` is a placeholder).

## Requirements

- Python 3.11+
- No external runtime dependencies — the game itself is pure standard library.
  `make` sets up an isolated `.venv` for the dev tools (pytest, mypy, ruff) since
  this system's Python is externally managed and won't allow global `pip install`.

## Usage

```bash
make install     # create .venv and install dev dependencies
make test        # run the pytest suite
make typecheck    # run mypy in strict mode
make lint         # run ruff
make check        # lint + typecheck + test — run this before committing
make run-board    # render a sample board (board.py's demo)
make run-engine   # run a scripted engine demo (moves + wall placement)
make clean        # remove the venv and all caches
```

Run `make help` to see this list from the terminal.

## Type checking

Game code (`board.py`, `engine.py`, `brain_v1.py`) is fully type-hinted and checked
with `mypy --strict`. This is a direct response to a real bug we hit early on:
`QuoridorBoard.from_dict` was reassigning `h_walls`/`v_walls` (declared as
`set[tuple[int, int]]`) straight from JSON-shaped lists, so a restored board silently
carried lists where the rest of the code expected sets. The fix was twofold —
correct the bug, and make it structurally impossible to reintroduce quietly:
`to_dict`/`from_dict` now go through an explicit `BoardState` `TypedDict` instead of
an untyped `dict`, and `from_dict` assigns each field explicitly (no more generic
`setattr` loop) so mypy can actually catch a type mismatch on that boundary again.
`make check` (and ideally CI, once there is one) runs `mypy` on every change.

Test files under `tests/` are exempted from mypy's "must annotate every def" rules
(conventional for pytest-style tests) but still get everything else strict mode checks.
