from typing import TypedDict


class BoardState(TypedDict):
    size: int
    player_count: int
    positions: list[list[int]]
    walls_left: list[int]
    h_walls: list[tuple[int, int]]
    v_walls: list[tuple[int, int]]


size_to_walls_map: dict[int, int] = {
    5: 3,
    7: 7,
    9: 10
}

VALID_PLAYER_COUNTS: set[int] = {2, 4}


def start_position(player: int, size: int) -> list[int]:
    mid = size // 2
    if player == 1:
        return [0, mid]
    if player == 2:
        return [size - 1, mid]
    if player == 3:
        return [mid, 0]
    if player == 4:
        return [mid, size - 1]
    raise ValueError(f"invalid player: {player}")


def goal_cells(player: int, size: int) -> frozenset[tuple[int, int]]:
    if player == 1:
        return frozenset((size - 1, c) for c in range(size))
    if player == 2:
        return frozenset((0, c) for c in range(size))
    if player == 3:
        return frozenset((r, size - 1) for r in range(size))
    if player == 4:
        return frozenset((r, 0) for r in range(size))
    raise ValueError(f"invalid player: {player}")


def size_to_walls(size: int, player_count: int = 2) -> int:
    if size not in size_to_walls_map:
        raise ValueError("size is invalid")
    if player_count not in VALID_PLAYER_COUNTS:
        raise ValueError("player_count is invalid")
    # Assumes an even player_count (VALID_PLAYER_COUNTS is what actually
    # guards that) — halving the 2-player pool per extra pair of seats.
    wall_count = size_to_walls_map.get(size)
    if not wall_count:
        raise RuntimeError("wall count should evaluate to true")
    return wall_count // (player_count // 2)


class QuoridorBoard:
    def __init__(self, size: int = 9, player_count: int = 2):
        if size not in size_to_walls_map:
            raise ValueError("size is invalid")
        if player_count not in VALID_PLAYER_COUNTS:
            raise ValueError("player_count is invalid")
        self.size: int = size
        self.player_count: int = player_count

        self.positions: list[list[int]] = [
            start_position(player, size) for player in range(1, player_count + 1)
        ]

        # Wall Slots:
        # h_walls: (row, col) wall placed directly below cell (row, col)
        # v_walls: (row, col) wall placed directly to the right of cell (row, col)
        self.h_walls: set[tuple[int, int]] = set()
        self.v_walls: set[tuple[int, int]] = set()

        self.walls_left: list[int] = [size_to_walls(size, player_count)] * player_count

    def to_dict(self) -> BoardState:
        # Every field here must be a fresh copy, not a reference into this
        # board's own mutable state — any caller that keeps a BoardState
        # around past a later move/wall placement (e.g. recorded game
        # history) would otherwise see it silently mutate out from under
        # them, since self.positions/self.walls_left are the exact same
        # list objects the engine keeps mutating in place.
        return {
            "size": self.size,
            "player_count": self.player_count,
            "positions": [list(pos) for pos in self.positions],
            "walls_left": list(self.walls_left),
            "h_walls": list(self.h_walls),
            "v_walls": list(self.v_walls),
        }

    @classmethod
    def from_dict(cls, data: BoardState) -> "QuoridorBoard":
        instance = cls(data["size"], data["player_count"])
        instance.positions = [list(pos) for pos in data["positions"]]
        instance.walls_left = list(data["walls_left"])
        instance.h_walls = set(data["h_walls"])
        instance.v_walls = set(data["v_walls"])
        return instance

    def get_state(self) -> BoardState:
        return self.to_dict()
