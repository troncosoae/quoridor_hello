import hashlib

from quoridor.board import BoardState


def state_key(state: BoardState, current_player: int) -> str:
    """Canonical, deterministic hash of a board state + whose turn it is.
    Two calls with an equal (state, current_player) always produce the
    same key, regardless of set-iteration order in h_walls/v_walls.

    Sorting h_walls/v_walls before hashing is load-bearing, not cosmetic:
    QuoridorBoard.to_dict() returns list(self.h_walls)/list(self.v_walls)
    straight off Python sets with no ordering guarantee, so two
    structurally-identical boards reached via different move orders could
    otherwise hash to different keys. positions/walls_left are safe as-is
    — plain lists indexed by player number, never set-derived.

    current_player is part of the key deliberately: the same raw board
    means something different depending on whose turn it is.
    """
    positions = tuple(tuple(p) for p in state["positions"])
    h_walls = tuple(sorted(tuple(w) for w in state["h_walls"]))
    v_walls = tuple(sorted(tuple(w) for w in state["v_walls"]))
    walls_left = tuple(state["walls_left"])
    canonical = (
        state["size"], state["player_count"], current_player,
        positions, h_walls, v_walls, walls_left,
    )
    return hashlib.sha256(repr(canonical).encode()).hexdigest()
