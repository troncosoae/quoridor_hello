import pytest

from quoridor.board import QuoridorBoard
from quoridor.rendering import CLIRenderer, Renderer


class TestRenderer:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Renderer()  # type: ignore[abstract]


class TestCLIRenderer:
    def test_render_is_pure(self, capsys):
        board = QuoridorBoard(5)
        CLIRenderer().render(board.get_state())
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_render_shows_pawn_markers(self):
        board = QuoridorBoard(5)
        output = CLIRenderer().render(board.get_state())
        assert "1" in output
        assert "2" in output

    def test_render_shows_wall_inventory(self):
        board = QuoridorBoard(5)
        output = CLIRenderer().render(board.get_state())
        assert "P1 walls left: 3" in output
        assert "P2 walls left: 3" in output

    def test_render_shows_horizontal_wall(self):
        board = QuoridorBoard(5)
        board.h_walls.add((2, 2))
        output = CLIRenderer().render(board.get_state())
        assert "---" in output

    def test_render_shows_vertical_wall(self):
        board = QuoridorBoard(5)
        board.v_walls.add((2, 2))
        output = CLIRenderer().render(board.get_state())
        assert "|" in output

    def test_render_returns_correct_row_count(self):
        board = QuoridorBoard(5)
        output = CLIRenderer().render(board.get_state())
        # header + (2*size - 1) grid rows + wall-count line
        assert len(output.splitlines()) == 1 + (2 * 5 - 1) + 1

    def test_render_shows_all_four_players(self):
        board = QuoridorBoard(5, player_count=4)
        output = CLIRenderer().render(board.get_state())
        for marker in ("1", "2", "3", "4"):
            assert marker in output
        for i in range(1, 5):
            assert f"P{i} walls left: 1" in output
