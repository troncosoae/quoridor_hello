import pytest

from board import QuoridorBoard, size_to_walls


class TestQuoridorBoard:
    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            QuoridorBoard(size=4)

    @pytest.mark.parametrize("size,expected_walls", [(5, 3), (7, 7), (9, 10)])
    def test_wall_inventory_by_size(self, size, expected_walls):
        board = QuoridorBoard(size)
        assert board.p1_walls_left == expected_walls
        assert board.p2_walls_left == expected_walls
        assert size_to_walls(size) == expected_walls

    def test_initial_pawn_positions(self):
        board = QuoridorBoard(9)
        assert board.p1_pos == [0, 4]
        assert board.p2_pos == [8, 4]

    def test_to_dict_from_dict_round_trip(self):
        board = QuoridorBoard(5)
        board.h_walls.add((0, 0))
        board.v_walls.add((2, 2))
        board.p1_walls_left = 1

        restored = QuoridorBoard.from_dict(board.to_dict())

        assert restored.size == board.size
        assert restored.p1_pos == board.p1_pos
        assert restored.p2_pos == board.p2_pos
        assert restored.h_walls == board.h_walls
        assert restored.v_walls == board.v_walls
        assert restored.p1_walls_left == board.p1_walls_left
