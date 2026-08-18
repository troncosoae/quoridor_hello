import pytest

from quoridor.board import QuoridorBoard, goal_cells, size_to_walls, start_position


class TestQuoridorBoard:
    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            QuoridorBoard(size=4)

    def test_invalid_player_count_raises(self):
        with pytest.raises(ValueError):
            QuoridorBoard(size=9, player_count=3)

    @pytest.mark.parametrize("size,expected_walls", [(5, 3), (7, 7), (9, 10)])
    def test_wall_inventory_by_size_two_players(self, size, expected_walls):
        board = QuoridorBoard(size)
        assert board.walls_left == [expected_walls, expected_walls]
        assert size_to_walls(size) == expected_walls

    @pytest.mark.parametrize("size,expected_walls", [(5, 1), (7, 3), (9, 5)])
    def test_wall_inventory_by_size_four_players(self, size, expected_walls):
        board = QuoridorBoard(size, player_count=4)
        assert board.walls_left == [expected_walls] * 4
        assert size_to_walls(size, player_count=4) == expected_walls

    def test_initial_pawn_positions_two_players(self):
        board = QuoridorBoard(9)
        assert board.positions == [[0, 4], [8, 4]]

    def test_initial_pawn_positions_four_players(self):
        board = QuoridorBoard(9, player_count=4)
        assert board.positions == [[0, 4], [8, 4], [4, 0], [4, 8]]

    def test_to_dict_from_dict_round_trip(self):
        board = QuoridorBoard(5, player_count=4)
        board.h_walls.add((0, 0))
        board.v_walls.add((2, 2))
        board.walls_left[0] = 0

        restored = QuoridorBoard.from_dict(board.to_dict())

        assert restored.size == board.size
        assert restored.player_count == board.player_count
        assert restored.positions == board.positions
        assert restored.h_walls == board.h_walls
        assert restored.v_walls == board.v_walls
        assert restored.walls_left == board.walls_left

    def test_from_dict_does_not_alias_caller_s_positions(self):
        board = QuoridorBoard(5)
        data = board.to_dict()

        restored = QuoridorBoard.from_dict(data)
        restored.positions[0][0] = 99

        assert data["positions"][0][0] != 99


class TestStartPosition:
    @pytest.mark.parametrize(
        "player,expected", [(1, [0, 4]), (2, [8, 4]), (3, [4, 0]), (4, [4, 8])]
    )
    def test_start_position(self, player, expected):
        assert start_position(player, size=9) == expected

    def test_invalid_player_raises(self):
        with pytest.raises(ValueError):
            start_position(5, size=9)


class TestGoalCells:
    def test_player_one_goal_is_bottom_row(self):
        cells = goal_cells(1, size=5)
        assert cells == frozenset((4, c) for c in range(5))

    def test_player_two_goal_is_top_row(self):
        cells = goal_cells(2, size=5)
        assert cells == frozenset((0, c) for c in range(5))

    def test_player_three_goal_is_rightmost_column(self):
        cells = goal_cells(3, size=5)
        assert cells == frozenset((r, 4) for r in range(5))

    def test_player_four_goal_is_leftmost_column(self):
        cells = goal_cells(4, size=5)
        assert cells == frozenset((r, 0) for r in range(5))

    def test_invalid_player_raises(self):
        with pytest.raises(ValueError):
            goal_cells(0, size=5)
