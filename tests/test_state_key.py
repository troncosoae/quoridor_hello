from quoridor.board import QuoridorBoard
from quoridor.state_key import state_key


def make_state(size=5, player_count=2):
    return QuoridorBoard(size, player_count).to_dict()


def test_identical_states_produce_identical_keys():
    state = make_state()
    assert state_key(state, 1) == state_key(state, 1)


def test_current_player_is_part_of_the_key():
    state = make_state()
    assert state_key(state, 1) != state_key(state, 2)


def test_wall_iteration_order_does_not_affect_the_key():
    state_a = make_state()
    state_a["h_walls"] = [(0, 0), (1, 1)]
    state_b = make_state()
    state_b["h_walls"] = [(1, 1), (0, 0)]

    assert state_key(state_a, 1) == state_key(state_b, 1)


def test_different_walls_produce_different_keys():
    state_a = make_state()
    state_a["h_walls"] = [(0, 0)]
    state_b = make_state()
    state_b["h_walls"] = [(0, 1)]

    assert state_key(state_a, 1) != state_key(state_b, 1)


def test_different_positions_produce_different_keys():
    state_a = make_state()
    state_b = make_state()
    state_b["positions"] = [[1, 2], state_b["positions"][1]]

    assert state_key(state_a, 1) != state_key(state_b, 1)
