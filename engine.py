from collections import deque
from enum import Enum

from board import QuoridorBoard


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


class QuoridorEngine:
    def __init__(self, board: QuoridorBoard | None = None):
        self.board = board if board is not None else QuoridorBoard()
        self.current_player: int = 1

    def _pawn_pos(self, player: int) -> list[int]:
        return self.board.p1_pos if player == 1 else self.board.p2_pos

    def _other_pawn_pos(self, player: int) -> list[int]:
        return self.board.p2_pos if player == 1 else self.board.p1_pos

    def _is_wall_between(self, row: int, col: int, new_row: int, new_col: int) -> bool:
        if new_row == row + 1:  # moving down
            return (row, col) in self.board.h_walls or (row, col - 1) in self.board.h_walls
        if new_row == row - 1:  # moving up
            return (new_row, col) in self.board.h_walls or (new_row, col - 1) in self.board.h_walls
        if new_col == col + 1:  # moving right
            return (row, col) in self.board.v_walls or (row - 1, col) in self.board.v_walls
        if new_col == col - 1:  # moving left
            return (row, new_col) in self.board.v_walls or (row - 1, new_col) in self.board.v_walls
        return False

    def is_valid_move(self, player: int, direction: Direction) -> bool:
        row, col = self._pawn_pos(player)
        d_row, d_col = _DIRECTION_DELTAS[direction]
        new_row, new_col = row + d_row, col + d_col

        if not (0 <= new_row < self.board.size and 0 <= new_col < self.board.size):
            return False

        if self._is_wall_between(row, col, new_row, new_col):
            return False

        if [new_row, new_col] == self._other_pawn_pos(player):
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

        self.current_player = 2 if player == 1 else 1

    def _walls_left(self, player: int) -> int:
        return self.board.p1_walls_left if player == 1 else self.board.p2_walls_left

    def _decrement_walls_left(self, player: int) -> None:
        if player == 1:
            self.board.p1_walls_left -= 1
        else:
            self.board.p2_walls_left -= 1

    def _wall_set(self, orientation: WallOrientation) -> set[tuple[int, int]]:
        if orientation == WallOrientation.HORIZONTAL:
            return self.board.h_walls
        return self.board.v_walls

    def _has_path_to_goal(self, player: int) -> bool:
        pawn_row, pawn_col = self._pawn_pos(player)
        start: tuple[int, int] = (pawn_row, pawn_col)
        goal_row = self.board.size - 1 if player == 1 else 0

        visited = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            if row == goal_row:
                return True
            for d_row, d_col in _DIRECTION_DELTAS.values():
                new_row, new_col = row + d_row, col + d_col
                if not (0 <= new_row < self.board.size and 0 <= new_col < self.board.size):
                    continue
                if self._is_wall_between(row, col, new_row, new_col):
                    continue
                if (new_row, new_col) in visited:
                    continue
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))
        return False

    def is_valid_wall_placement(
        self, player: int, orientation: WallOrientation, row: int, col: int
    ) -> bool:
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
        both_have_path = self._has_path_to_goal(1) and self._has_path_to_goal(2)
        wall_set.discard((row, col))

        return both_have_path

    def place_wall(self, player: int, orientation: WallOrientation, row: int, col: int) -> None:
        if player != self.current_player:
            raise InvalidMoveError(f"It is player {self.current_player}'s turn")

        if not self.is_valid_wall_placement(player, orientation, row, col):
            raise InvalidMoveError(
                f"Invalid wall placement: {orientation.value} at ({row}, {col}) for player {player}"
            )

        self._wall_set(orientation).add((row, col))
        self._decrement_walls_left(player)
        self.current_player = 2 if player == 1 else 1

    def is_won(self, player: int) -> bool:
        row, _ = self._pawn_pos(player)
        return row == self.board.size - 1 if player == 1 else row == 0

    def winner(self) -> int | None:
        if self.is_won(1):
            return 1
        if self.is_won(2):
            return 2
        return None


if __name__ == "__main__":
    from board import CLIRenderer

    def try_place_wall(
        engine: QuoridorEngine, player: int, orientation: WallOrientation, row: int, col: int
    ) -> None:
        try:
            engine.place_wall(player, orientation, row, col)
            print(f"\nPlayer {player} placed a {orientation.value} wall at ({row}, {col})")
        except InvalidMoveError as e:
            print(f"\nPlayer {player} wall placement rejected: {e}")
        CLIRenderer.render(engine.board)

    def try_move(engine: QuoridorEngine, player: int, direction: Direction) -> None:
        try:
            engine.move(player, direction)
            print(f"\nPlayer {player} moved {direction.value}")
        except InvalidMoveError as e:
            print(f"\nPlayer {player} move {direction.value} rejected: {e}")
        CLIRenderer.render(engine.board)

    engine = QuoridorEngine(QuoridorBoard(5))
    CLIRenderer.render(engine.board)

    try_place_wall(engine, 1, WallOrientation.HORIZONTAL, 0, 1)
    try_place_wall(engine, 2, WallOrientation.HORIZONTAL, 0, 1)  # rejected: overlap
    try_move(engine, 2, Direction.LEFT)
