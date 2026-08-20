import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from quoridor.actions import Action, action_from_dict, action_to_dict
from quoridor.board import BoardState


@dataclass
class GamePly:
    state: BoardState
    current_player: int
    action: Action
    actor: str
    """Agent.KIND at the time this ply was played — e.g. "model" vs "bfs"
    matters for policy-target derivation (quoridor.rl.targets)."""
    legal_actions: list[Action]
    """Captured once when the ply was recorded, not recomputed later —
    avoids reconstructing a throwaway engine per stored ply at training
    time just to re-derive this."""


@dataclass
class GameRecord:
    game_id: str
    batch_index: int
    size: int
    player_count: int
    plies: list[GamePly]
    winner: int | None
    """None means the game hit a ply cutoff with no winner — excluded from
    target derivation."""


class GameStore(ABC):
    """Where game histories live — decoupled from both game execution
    (GameRunner records into one of these, optionally) and the training
    algorithm (which only ever reads from one). Swappable storage
    mechanism: in-memory and a plain text (JSON-lines) file for now,
    MongoDB later without changing either of those callers."""

    @abstractmethod
    def save_game(self, record: GameRecord) -> None: ...

    @abstractmethod
    def get_game(self, game_id: str) -> GameRecord | None: ...

    @abstractmethod
    def games_in_window(self, lookback_batches: int, upto_batch: int) -> list[GameRecord]:
        """Every stored game whose batch_index falls within the last
        `lookback_batches` batches, up to and including `upto_batch`."""
        ...


def _record_to_dict(record: GameRecord) -> dict[str, Any]:
    return {
        "game_id": record.game_id,
        "batch_index": record.batch_index,
        "size": record.size,
        "player_count": record.player_count,
        "winner": record.winner,
        "plies": [
            {
                "state": dict(ply.state),
                "current_player": ply.current_player,
                "action": action_to_dict(ply.action),
                "actor": ply.actor,
                "legal_actions": [action_to_dict(a) for a in ply.legal_actions],
            }
            for ply in record.plies
        ],
    }


def _record_from_dict(data: dict[str, Any]) -> GameRecord:
    return GameRecord(
        game_id=data["game_id"],
        batch_index=data["batch_index"],
        size=data["size"],
        player_count=data["player_count"],
        winner=data["winner"],
        plies=[
            GamePly(
                state=ply["state"],
                current_player=ply["current_player"],
                action=action_from_dict(ply["action"]),
                actor=ply["actor"],
                legal_actions=[action_from_dict(a) for a in ply["legal_actions"]],
            )
            for ply in data["plies"]
        ],
    )


class InMemoryGameStore(GameStore):
    """Holds every game in a plain list. Known, accepted limitation: never
    evicts games that have aged out of every possible future window — fine
    at low-thousands-of-games scale, genuinely leaky in RAM at the ~1M
    games eventually intended. A MongoDB-backed store is the long-term fix
    for that, not built here."""

    def __init__(self) -> None:
        self._games: list[GameRecord] = []

    def save_game(self, record: GameRecord) -> None:
        self._games.append(record)

    def get_game(self, game_id: str) -> GameRecord | None:
        return next((g for g in self._games if g.game_id == game_id), None)

    def games_in_window(self, lookback_batches: int, upto_batch: int) -> list[GameRecord]:
        min_batch = max(0, upto_batch - lookback_batches + 1)
        return [g for g in self._games if min_batch <= g.batch_index <= upto_batch]


class FileGameStore(GameStore):
    """One JSON object per line, appended as each game finishes — a
    genuinely plain text file, just structured per line. Re-reads the
    whole file on every query (same "recompute fresh" philosophy as
    InMemoryGameStore, same acknowledged scaling limit — a real index, or
    the eventual MongoDB swap, is the fix, not built here)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def save_game(self, record: GameRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(_record_to_dict(record)) + "\n")

    def get_game(self, game_id: str) -> GameRecord | None:
        return next((r for r in self._read_all() if r.game_id == game_id), None)

    def games_in_window(self, lookback_batches: int, upto_batch: int) -> list[GameRecord]:
        min_batch = max(0, upto_batch - lookback_batches + 1)
        return [r for r in self._read_all() if min_batch <= r.batch_index <= upto_batch]

    def _read_all(self) -> Iterator[GameRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield _record_from_dict(json.loads(line))
