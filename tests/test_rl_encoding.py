import pytest

from quoridor.actions import MoveAction, WallAction
from quoridor.board import QuoridorBoard
from quoridor.engine import Direction, QuoridorEngine, WallOrientation
from quoridor.rl.encoding import (
    action_size,
    action_to_index,
    encode_state,
    index_to_action,
    legal_action_mask,
    num_planes,
)


def make_engine(size=5, player_count=2):
    return QuoridorEngine(QuoridorBoard(size, player_count))


class TestActionSizeAndPlanes:
    @pytest.mark.parametrize(
        "size,expected", [(5, 4 + 2 * 4 * 4), (7, 4 + 2 * 6 * 6), (9, 4 + 2 * 8 * 8)]
    )
    def test_action_size_formula(self, size, expected):
        assert action_size(size) == expected

    @pytest.mark.parametrize("player_count,expected", [(2, 8), (4, 14)])
    def test_num_planes_formula(self, player_count, expected):
        assert num_planes(player_count) == expected


class TestActionIndexRoundTrip:
    @pytest.mark.parametrize("size", [5, 9])
    def test_every_index_round_trips(self, size):
        for index in range(action_size(size)):
            action = index_to_action(index, size)
            assert action_to_index(action, size) == index

    def test_move_actions_map_to_the_first_four_indices(self, size=9):
        for direction in Direction:
            index = action_to_index(MoveAction(direction), size)
            assert 0 <= index < 4

    def test_wall_actions_map_to_indices_after_moves(self, size=5):
        index = action_to_index(WallAction(WallOrientation.HORIZONTAL, 0, 0), size)
        assert index >= 4
        index = action_to_index(WallAction(WallOrientation.VERTICAL, size - 2, size - 2), size)
        assert index == action_size(size) - 1


class TestEncodeState:
    def test_shape_matches_num_planes_and_size(self):
        engine = make_engine(5, player_count=2)
        state = engine.get_state()
        x = encode_state(state, engine.current_player)
        assert tuple(x.shape) == (num_planes(2), 5, 5)

    def test_pawn_positions_are_one_hot(self):
        engine = make_engine(5, player_count=2)
        state = engine.get_state()
        x = encode_state(state, engine.current_player)
        p1_row, p1_col = state["positions"][0]
        p2_row, p2_col = state["positions"][1]
        assert x[0, p1_row, p1_col] == 1.0
        assert x[1, p2_row, p2_col] == 1.0

    def test_wall_planes_reflect_placed_walls(self):
        engine = make_engine(5, player_count=2)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        state = engine.get_state()
        x = encode_state(state, engine.current_player)
        h_channel = 2  # after 2 pawn-position planes for player_count=2
        assert x[h_channel, 0, 0] == 1.0


class TestLegalActionMask:
    def test_agrees_with_engine_is_valid_move(self):
        engine = make_engine(5, player_count=2)
        size = engine.get_state()["size"]
        mask = legal_action_mask(engine, 1, size)
        for direction in Direction:
            index = action_to_index(MoveAction(direction), size)
            assert bool(mask[index]) == engine.is_valid_move(1, direction)

    def test_agrees_with_engine_is_valid_wall_placement(self):
        engine = make_engine(5, player_count=2)
        engine.place_wall(1, WallOrientation.HORIZONTAL, 0, 0)
        # It's player 2's turn now; check a wall slot known to be illegal
        # (overlaps the wall player 1 just placed) and one that's legal.
        size = engine.get_state()["size"]
        mask = legal_action_mask(engine, 2, size)

        overlap_index = action_to_index(WallAction(WallOrientation.HORIZONTAL, 0, 1), size)
        assert bool(mask[overlap_index]) == engine.is_valid_wall_placement(
            2, WallOrientation.HORIZONTAL, 0, 1
        )

        clear_index = action_to_index(WallAction(WallOrientation.VERTICAL, 2, 2), size)
        assert bool(mask[clear_index]) == engine.is_valid_wall_placement(
            2, WallOrientation.VERTICAL, 2, 2
        )
