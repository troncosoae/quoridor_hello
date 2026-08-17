import pytest

from board import QuoridorBoard
from engine import Direction, InvalidMoveError, QuoridorEngine, WallOrientation


def make_engine(size=5):
    return QuoridorEngine(QuoridorBoard(size))


class TestMovement:
    def test_initial_positions(self):
        engine = make_engine(5)
        assert engine.board.p1_pos == [0, 2]
        assert engine.board.p2_pos == [4, 2]

    def test_valid_move_updates_position_and_turn(self):
        engine = make_engine(5)
        engine.move(1, Direction.DOWN)
        assert engine.board.p1_pos == [1, 2]
        assert engine.current_player == 2

    def test_move_out_of_turn_raises(self):
        engine = make_engine(5)
        with pytest.raises(InvalidMoveError):
            engine.move(2, Direction.UP)

    def test_move_off_board_raises(self):
        engine = make_engine(5)
        with pytest.raises(InvalidMoveError):
            engine.move(1, Direction.UP)

    def test_move_onto_opponent_raises(self):
        engine = make_engine(5)
        engine.move(1, Direction.DOWN)  # p1 -> (1, 2)
        engine.move(2, Direction.UP)  # p2 -> (3, 2)
        engine.move(1, Direction.DOWN)  # p1 -> (2, 2)
        assert engine.board.p1_pos == [2, 2]
        assert engine.board.p2_pos == [3, 2]
        with pytest.raises(InvalidMoveError):
            engine.move(2, Direction.UP)

    def test_move_blocked_by_horizontal_wall(self):
        engine = make_engine(5)
        engine.board.h_walls.add((0, 2))
        with pytest.raises(InvalidMoveError):
            engine.move(1, Direction.DOWN)

    def test_move_blocked_by_vertical_wall(self):
        engine = make_engine(5)
        engine.board.v_walls.add((0, 2))
        with pytest.raises(InvalidMoveError):
            engine.move(1, Direction.RIGHT)

    def test_horizontal_wall_blocks_both_columns_it_spans(self):
        engine = make_engine(5)
        engine.board.h_walls.add((0, 1))
        with pytest.raises(InvalidMoveError):
            engine.move(1, Direction.DOWN)  # p1 at col 2, wall spans cols 1-2


class TestWinCondition:
    def test_not_won_at_start(self):
        engine = make_engine(5)
        assert engine.winner() is None

    def test_player_one_wins_on_far_row(self):
        engine = make_engine(5)
        engine.board.p1_pos = [4, 2]
        assert engine.is_won(1) is True
        assert engine.winner() == 1

    def test_player_two_wins_on_near_row(self):
        engine = make_engine(5)
        engine.board.p2_pos = [0, 2]
        assert engine.is_won(2) is True
        assert engine.winner() == 2


class TestWallPlacement:
    def test_place_wall_decrements_count_and_switches_turn(self):
        engine = make_engine(5)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        assert (0, 0) in engine.board.h_walls
        assert engine.board.p1_walls_left == 2
        assert engine.current_player == 2

    def test_wall_out_of_turn_raises(self):
        engine = make_engine(5)
        with pytest.raises(InvalidMoveError):
            engine.place_wall(2, WallOrientation.HORIZONTAL, 0, 0)

    def test_duplicate_wall_rejected(self):
        engine = make_engine(5)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        with pytest.raises(InvalidMoveError):
            engine.place_wall(2, WallOrientation.HORIZONTAL, 0, 0)

    def test_overlapping_horizontal_walls_rejected(self):
        engine = make_engine(5)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        with pytest.raises(InvalidMoveError):
            engine.place_wall(2, WallOrientation.HORIZONTAL, 0, 1)

    def test_overlapping_vertical_walls_rejected(self):
        engine = make_engine(5)
        engine.place_wall(1, WallOrientation.VERTICAL, 0, 0)
        with pytest.raises(InvalidMoveError):
            engine.place_wall(2, WallOrientation.VERTICAL, 1, 0)

    def test_crossing_walls_rejected(self):
        engine = make_engine(5)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 2, 2)
        with pytest.raises(InvalidMoveError):
            engine.place_wall(2, WallOrientation.VERTICAL, 2, 2)

    def test_out_of_bounds_wall_rejected(self):
        engine = make_engine(5)
        with pytest.raises(InvalidMoveError):
            engine.place_wall(1, WallOrientation.HORIZONTAL, 4, 0)
        with pytest.raises(InvalidMoveError):
            engine.place_wall(1, WallOrientation.VERTICAL, 0, 4)

    def test_wall_fully_blocking_path_rejected(self):
        engine = make_engine(5)
        engine.move(1, Direction.LEFT)  # p1 -> (0, 1)
        engine.move(2, Direction.LEFT)  # p2 -> (4, 1)
        engine.move(1, Direction.LEFT)  # p1 -> (0, 0)
        engine.current_player = 1
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        engine.current_player = 1
        with pytest.raises(InvalidMoveError):
            engine.place_wall(1, WallOrientation.VERTICAL, 0, 0)

    def test_wall_inventory_exhausted(self):
        engine = make_engine(5)  # 3 walls per player on a 5x5 board
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        engine.place_wall(2, WallOrientation.HORIZONTAL, 0, 2)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 1, 0)
        engine.place_wall(2, WallOrientation.HORIZONTAL, 1, 2)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 2, 0)
        engine.place_wall(2, WallOrientation.HORIZONTAL, 2, 2)

        assert engine.board.p1_walls_left == 0
        with pytest.raises(InvalidMoveError):
            engine.place_wall(1, WallOrientation.HORIZONTAL, 3, 0)
