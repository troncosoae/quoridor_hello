from quoridor.actions import MoveAction
from quoridor.engine import Direction
from quoridor.game_store import GamePly, GameRecord
from quoridor.rl.targets import derive_training_targets
from quoridor.state_key import state_key


def make_state(positions):
    return {
        "size": 5,
        "player_count": 2,
        "positions": positions,
        "walls_left": [3, 3],
        "h_walls": [],
        "v_walls": [],
    }


S0 = make_state([[0, 2], [4, 2]])
S1 = make_state([[1, 2], [4, 2]])
S2 = make_state([[2, 2], [3, 2]])
S3 = make_state([[0, 2], [3, 2]])
S4 = make_state([[1, 2], [3, 2]])
S5 = make_state([[0, 2], [2, 2]])

DOWN, UP, LEFT, RIGHT = (
    MoveAction(Direction.DOWN),
    MoveAction(Direction.UP),
    MoveAction(Direction.LEFT),
    MoveAction(Direction.RIGHT),
)


def test_value_and_policy_derived_from_a_two_ply_win():
    game = GameRecord(
        game_id="g1", batch_index=0, size=5, player_count=2,
        plies=[
            GamePly(
                state=S0, current_player=1, action=DOWN,
                actor="model", legal_actions=[DOWN, UP],
            ),
            GamePly(
                state=S1, current_player=2, action=LEFT, actor="bfs", legal_actions=[LEFT]
            ),
        ],
        winner=1,
    )

    targets = derive_training_targets([game])

    s0_target = targets[state_key(S0, 1)]
    assert s0_target.value == [1.0, 0.0]
    assert s0_target.legal_actions == [DOWN, UP]
    assert s0_target.policy == {DOWN: 1.0}

    s1_target = targets[state_key(S1, 2)]
    assert s1_target.value == [1.0, 0.0]
    assert s1_target.policy == {}


def test_terminal_ply_uses_the_known_outcome_not_a_lookup():
    # A single-ply game: the winning move itself, actor="model". Its
    # resulting (terminal) state is never itself recorded as a ply, so the
    # successor value must come from game.winner directly.
    game = GameRecord(
        game_id="g2", batch_index=0, size=5, player_count=2,
        plies=[GamePly(state=S2, current_player=2, action=UP, actor="model", legal_actions=[UP])],
        winner=2,
    )

    targets = derive_training_targets([game])

    s2_target = targets[state_key(S2, 2)]
    assert s2_target.value == [0.0, 1.0]
    assert s2_target.policy == {UP: 1.0}


def test_zero_sum_policy_weights_fall_back_to_uniform():
    # Two model-played actions from the same state, both leading to a
    # certain loss for the mover — every sampled weight is 0.0, which would
    # otherwise divide by zero when normalizing.
    game_c = GameRecord(
        game_id="gc", batch_index=0, size=5, player_count=2,
        plies=[
            GamePly(
                state=S3, current_player=1, action=DOWN,
                actor="model", legal_actions=[DOWN, UP],
            ),
            GamePly(
                state=S4, current_player=2, action=LEFT, actor="bfs", legal_actions=[LEFT]
            ),
        ],
        winner=2,
    )
    game_d = GameRecord(
        game_id="gd", batch_index=0, size=5, player_count=2,
        plies=[
            GamePly(
                state=S3, current_player=1, action=UP,
                actor="model", legal_actions=[DOWN, UP],
            ),
            GamePly(
                state=S5, current_player=2, action=RIGHT, actor="bfs", legal_actions=[RIGHT]
            ),
        ],
        winner=2,
    )

    targets = derive_training_targets([game_c, game_d])

    s3_target = targets[state_key(S3, 1)]
    assert s3_target.policy == {DOWN: 0.5, UP: 0.5}


def test_games_without_a_winner_are_excluded():
    cutoff_game = GameRecord(
        game_id="g3", batch_index=0, size=5, player_count=2,
        plies=[
            GamePly(state=S0, current_player=1, action=DOWN, actor="model", legal_actions=[DOWN])
        ],
        winner=None,
    )

    targets = derive_training_targets([cutoff_game])

    assert targets == {}
