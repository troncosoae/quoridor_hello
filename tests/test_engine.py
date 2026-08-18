import time

import pytest

from quoridor import pathfinding
from quoridor.board import QuoridorBoard
from quoridor.engine import Direction, InvalidMoveError, QuoridorEngine, WallOrientation
from quoridor.timeouts import TimeoutExceededError


def make_engine(size=5, player_count=2):
    return QuoridorEngine(QuoridorBoard(size, player_count))


class TestMovement:
    def test_initial_positions(self):
        engine = make_engine(5)
        assert engine.board.positions[0] == [0, 2]
        assert engine.board.positions[1] == [4, 2]

    def test_valid_move_updates_position_and_turn(self):
        engine = make_engine(5)
        engine.move(1, Direction.DOWN)
        assert engine.board.positions[0] == [1, 2]
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
        assert engine.board.positions[0] == [2, 2]
        assert engine.board.positions[1] == [3, 2]
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

    def test_valid_move_out_of_range_player_returns_400able_error(self):
        engine = make_engine(5)
        with pytest.raises(ValueError):
            engine.is_valid_move(5, Direction.DOWN)

    def test_valid_wall_placement_out_of_range_player_returns_400able_error(self):
        engine = make_engine(5)
        with pytest.raises(ValueError):
            engine.is_valid_wall_placement(5, WallOrientation.HORIZONTAL, 0, 0)


class TestFourPlayerMovement:
    def test_initial_positions(self):
        engine = make_engine(5, player_count=4)
        assert engine.board.positions == [[0, 2], [4, 2], [2, 0], [2, 4]]

    def test_turn_cycles_through_all_four_players(self):
        engine = make_engine(9, player_count=4)
        # Move each player one step in a direction that never collides.
        engine.move(1, Direction.DOWN)
        assert engine.current_player == 2
        engine.move(2, Direction.UP)
        assert engine.current_player == 3
        engine.move(3, Direction.RIGHT)
        assert engine.current_player == 4
        engine.move(4, Direction.LEFT)
        assert engine.current_player == 1

    def test_move_blocked_by_any_other_players_pawn(self):
        engine = make_engine(9, player_count=4)
        engine.board.positions[1] = [1, 4]  # directly below p1's start
        with pytest.raises(InvalidMoveError):
            engine.move(1, Direction.DOWN)


class TestWinCondition:
    def test_not_won_at_start(self):
        engine = make_engine(5)
        assert engine.winner() is None

    def test_player_one_wins_on_far_row(self):
        engine = make_engine(5)
        engine.board.positions[0] = [4, 2]
        assert engine.is_won(1) is True
        assert engine.winner() == 1

    def test_player_two_wins_on_near_row(self):
        engine = make_engine(5)
        engine.board.positions[1] = [0, 2]
        assert engine.is_won(2) is True
        assert engine.winner() == 2

    def test_player_three_wins_on_rightmost_column(self):
        engine = make_engine(5, player_count=4)
        engine.board.positions[2] = [2, 4]
        assert engine.is_won(3) is True
        assert engine.winner() == 3

    def test_player_four_wins_on_leftmost_column(self):
        engine = make_engine(5, player_count=4)
        engine.board.positions[3] = [2, 0]
        assert engine.is_won(4) is True
        assert engine.winner() == 4


class TestWallPlacement:
    def test_place_wall_decrements_count_and_switches_turn(self):
        engine = make_engine(5)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        assert (0, 0) in engine.board.h_walls
        assert engine.board.walls_left[0] == 2
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

        assert engine.board.walls_left[0] == 0
        with pytest.raises(InvalidMoveError):
            engine.place_wall(1, WallOrientation.HORIZONTAL, 3, 0)

    def test_wall_check_timeout_does_not_leave_a_dangling_tentative_wall(self, monkeypatch):
        engine = make_engine(5)
        engine.WALL_CHECK_TIMEOUT_SECONDS = 0.01

        def slow_connected_components(h_walls, v_walls, size):
            time.sleep(0.2)
            return pathfinding.connected_components(h_walls, v_walls, size)

        monkeypatch.setattr(pathfinding, "connected_components", slow_connected_components)

        with pytest.raises(TimeoutExceededError):
            engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)

        assert engine.board.h_walls == set()
        assert engine.board.walls_left[0] == 3
        assert engine.current_player == 1


class TestFourPlayerWallPlacement:
    def test_wall_rejected_if_it_would_box_in_any_of_the_four_players(self):
        engine = make_engine(5, player_count=4)
        # Trap player 3 (starts at (2, 0)) in the corner (0, 0) area is
        # awkward to set up without moving it there first, so instead trap
        # it at its own start: walls below and to the right of (2, 0).
        engine.board.positions[2] = [0, 0]
        engine.current_player = 1
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        engine.current_player = 1
        with pytest.raises(InvalidMoveError):
            engine.place_wall(1, WallOrientation.VERTICAL, 0, 0)
