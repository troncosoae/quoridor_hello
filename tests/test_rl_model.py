import pytest

from quoridor.actions import MoveAction
from quoridor.board import QuoridorBoard
from quoridor.engine import QuoridorEngine
from quoridor.rl.cnn_model import CNNModel
from quoridor.rl.encoding import action_to_index
from quoridor.rl.model import Model


def make_engine(size=5, player_count=2):
    return QuoridorEngine(QuoridorBoard(size, player_count))


class TestModelABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Model()  # type: ignore[abstract]


class TestCNNModel:
    def test_declares_singleton_support(self):
        model = CNNModel(size=5, player_count=2)
        assert model.SUPPORTED_PLAYER_COUNTS == frozenset({2})
        assert model.SUPPORTED_BOARD_SIZES == frozenset({5})

    def test_predict_policy_covers_exactly_the_legal_actions(self):
        engine = make_engine(5, player_count=2)
        model = CNNModel(size=5, player_count=2)
        state = engine.get_state()

        prediction = model.predict(engine, 1, state)

        for action in prediction.policy:
            index = action_to_index(action, 5)
            if isinstance(action, MoveAction):
                assert engine.is_valid_move(1, action.direction)
            else:
                assert engine.is_valid_wall_placement(1, action.orientation, action.row, action.col)
            assert 0 <= index < 36

    def test_predict_policy_sums_to_roughly_one(self):
        engine = make_engine(5, player_count=2)
        model = CNNModel(size=5, player_count=2)
        prediction = model.predict(engine, 1, engine.get_state())
        assert sum(prediction.policy.values()) == pytest.approx(1.0, abs=1e-4)

    def test_predict_value_has_one_entry_per_player_and_sums_to_one(self):
        engine = make_engine(5, player_count=4)
        model = CNNModel(size=5, player_count=4)
        prediction = model.predict(engine, 1, engine.get_state())
        assert len(prediction.value) == 4
        assert sum(prediction.value) == pytest.approx(1.0, abs=1e-4)
