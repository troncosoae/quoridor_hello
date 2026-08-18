from enum import Enum
from typing import Protocol

from quoridor import pathfinding
from quoridor.board import BoardState, QuoridorBoard, goal_cells
from quoridor.timeouts import run_with_timeout


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class WallOrientation(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


_DIRECTION_DELTAS = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


class InvalidMoveError(Exception):
    pass


class EngineLike(Protocol):
    """The interface Agent/GameRunner/Renderer code is written against.

    QuoridorEngine (local, in-process) and RemoteEngine (quoridor.client,
    talks to a quoridor.server over HTTP) both satisfy this structurally.
    Code written against EngineLike must never reach past it (e.g. a local
    QuoridorEngine's `.board` attribute) — RemoteEngine has no such
    attribute, so anything that does breaks remote play silently.
    """

    current_player: int

    def get_state(self) -> BoardState: ...

    def is_valid_move(self, player: int, direction: Direction) -> bool: ...

    def move(self, player: int, direction: Direction) -> None: ...

    def is_valid_wall_placement(
        self, player: int, orientation: WallOrientation, row: int, col: int
    ) -> bool: ...

    def place_wall(
        self, player: int, orientation: WallOrientation, row: int, col: int
    ) -> None: ...

    def winner(self) -> int | None: ...


class QuoridorEngine:
    # Only wall placement runs open-ended search (a connected-components scan
    # to check no player gets fully boxed in); plain moves are O(1) and
    # don't need a budget. Defensive — see TimeoutExceededError's docstring.
    WALL_CHECK_TIMEOUT_SECONDS = 3.0

    def __init__(self, board: QuoridorBoard | None = None):
        self.board = board if board is not None else QuoridorBoard()
        self.current_player: int = 1

    def get_state(self) -> BoardState:
        return self.board.get_state()

    def _validate_player(self, player: int) -> None:
        if not (1 <= player <= self.board.player_count):
            raise ValueError(f"invalid player: {player}")

    def _pawn_pos(self, player: int) -> list[int]:
        return self.board.positions[player - 1]

    def _is_occupied_by_other(self, row: int, col: int, excluding_player: int) -> bool:
        for other in range(1, self.board.player_count + 1):
            if other != excluding_player and self.board.positions[other - 1] == [row, col]:
                return True
        return False

    def _is_wall_between(self, row: int, col: int, new_row: int, new_col: int) -> bool:
        return pathfinding.is_wall_between(
            self.board.h_walls, self.board.v_walls, row, col, new_row, new_col
        )

    def is_valid_move(self, player: int, direction: Direction) -> bool:
        self._validate_player(player)
        row, col = self._pawn_pos(player)
        d_row, d_col = _DIRECTION_DELTAS[direction]
        new_row, new_col = row + d_row, col + d_col

        if not (0 <= new_row < self.board.size and 0 <= new_col < self.board.size):
            return False

        if self._is_wall_between(row, col, new_row, new_col):
            return False

        if self._is_occupied_by_other(new_row, new_col, player):
            return False

        return True

    def move(self, player: int, direction: Direction) -> None:
        if player != self.current_player:
            raise InvalidMoveError(f"It is player {self.current_player}'s turn")

        if not self.is_valid_move(player, direction):
            raise InvalidMoveError(f"Invalid move: {direction.value} for player {player}")

        d_row, d_col = _DIRECTION_DELTAS[direction]
        pawn = self._pawn_pos(player)
        pawn[0] += d_row
        pawn[1] += d_col

        self.current_player = player % self.board.player_count + 1

    def _walls_left(self, player: int) -> int:
        return self.board.walls_left[player - 1]

    def _decrement_walls_left(self, player: int) -> None:
        self.board.walls_left[player - 1] -= 1

    def _wall_set(self, orientation: WallOrientation) -> set[tuple[int, int]]:
        if orientation == WallOrientation.HORIZONTAL:
            return self.board.h_walls
        return self.board.v_walls

    def _walls_still_allow_every_player_a_path(self) -> bool:
        def check() -> bool:
            components = pathfinding.connected_components(
                self.board.h_walls, self.board.v_walls, self.board.size
            )
            for player in range(1, self.board.player_count + 1):
                row, col = self._pawn_pos(player)
                region = pathfinding.region_containing(components, (row, col))
                if region.isdisjoint(goal_cells(player, self.board.size)):
                    return False
            return True

        return run_with_timeout(check, self.WALL_CHECK_TIMEOUT_SECONDS)

    def is_valid_wall_placement(
        self, player: int, orientation: WallOrientation, row: int, col: int
    ) -> bool:
        self._validate_player(player)
        if self._walls_left(player) <= 0:
            return False

        max_index = self.board.size - 2
        if not (0 <= row <= max_index and 0 <= col <= max_index):
            return False

        if (row, col) in self.board.h_walls or (row, col) in self.board.v_walls:
            return False

        if orientation == WallOrientation.HORIZONTAL:
            if (row, col - 1) in self.board.h_walls or (row, col + 1) in self.board.h_walls:
                return False
        else:
            if (row - 1, col) in self.board.v_walls or (row + 1, col) in self.board.v_walls:
                return False

        wall_set = self._wall_set(orientation)
        wall_set.add((row, col))
        try:
            return self._walls_still_allow_every_player_a_path()
        finally:
            # Guaranteed even if the check above times out — a tentative
            # wall must never survive past this method either way.
            wall_set.discard((row, col))

    def place_wall(
        self, player: int, orientation: WallOrientation, row: int, col: int
    ) -> None:
        if player != self.current_player:
            raise InvalidMoveError(f"It is player {self.current_player}'s turn")

        if not self.is_valid_wall_placement(player, orientation, row, col):
            raise InvalidMoveError(
                f"Invalid wall placement: {orientation.value} at ({row}, {col}) for player {player}"
            )

        self._wall_set(orientation).add((row, col))
        self._decrement_walls_left(player)
        self.current_player = player % self.board.player_count + 1

    def is_won(self, player: int) -> bool:
        row, col = self._pawn_pos(player)
        return (row, col) in goal_cells(player, self.board.size)

    def winner(self) -> int | None:
        for player in range(1, self.board.player_count + 1):
            if self.is_won(player):
                return player
        return None
