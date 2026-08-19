import pytest

from quoridor.actions import Action, MoveAction, WallAction
from quoridor.board import QuoridorBoard
from quoridor.engine import Direction, EngineLike, QuoridorEngine, WallOrientation
from quoridor.rl.mcts import MCTS, _MCTSNode
from quoridor.rl.model import Model, ModelPrediction


def make_engine(size=5, player_count=2):
    return QuoridorEngine(QuoridorBoard(size, player_count))


def _enumerate_legal(engine: EngineLike, player: int, state) -> list[Action]:
    actions: list[Action] = [MoveAction(d) for d in Direction if engine.is_valid_move(player, d)]
    max_index = state["size"] - 2
    for orientation in WallOrientation:
        for row in range(max_index + 1):
            for col in range(max_index + 1):
                if engine.is_valid_wall_placement(player, orientation, row, col):
                    actions.append(WallAction(orientation, row, col))
    return actions


class _FakeModel(Model):
    """Uniform policy over legal actions, fixed value — deterministic and
    fast, isolating MCTS's own search logic from any real network."""

    SUPPORTED_PLAYER_COUNTS = frozenset({2})
    SUPPORTED_BOARD_SIZES = frozenset({5, 9})

    def __init__(self, value: list[float] | None = None):
        self.value = value if value is not None else [0.5, 0.5]
        self.calls: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    def predict(self, engine: EngineLike, player: int, state) -> ModelPrediction:
        self.calls.append((player, tuple(tuple(p) for p in state["positions"])))
        legal = _enumerate_legal(engine, player, state)
        prior = 1.0 / len(legal)
        return ModelPrediction(policy={a: prior for a in legal}, value=list(self.value))


class TestPUCTSelectionRegression:
    def test_picks_the_immediately_winning_move_over_a_uniformly_scored_alternative(self):
        # Exact scenario from the design review: player 1 one step from its
        # goal row, with a winning DOWN move available alongside several
        # non-winning legal moves. A FakeModel with a UNIFORM policy and a
        # fixed 0.5/0.5 value for non-terminal expansions means the only
        # thing that can differentiate children is the search's own value
        # bookkeeping — exactly what the PUCT formula bug corrupted (it
        # would systematically favor the move that's best for the
        # OPPONENT). This fails under the pre-fix formula and passes under
        # the fix.
        engine = make_engine(5, player_count=2)
        engine.board.positions[0] = [3, 2]  # one step from row 4 (player 1's goal)
        engine.board.positions[1] = [4, 0]  # out of the way — (4,2) must stay open to win into

        model = _FakeModel()
        mcts = MCTS(model, num_simulations=40)

        action = mcts.run(engine.get_state(), current_player=1)

        assert action == MoveAction(Direction.DOWN)

    def test_visit_counts_are_conserved(self):
        engine = make_engine(5, player_count=2)
        engine.board.positions[0] = [3, 2]

        model = _FakeModel()
        mcts = MCTS(model, num_simulations=25)
        mcts.run(engine.get_state(), current_player=1)

        # run() doesn't expose the root, so rebuild one more expansion the
        # same way run() does internally to inspect it — a fresh MCTS run
        # with the same inputs is deterministic (uniform policy, fixed
        # value, no randomness anywhere in this search).
        root = _MCTSNode(state=engine.get_state(), current_player=1, parent=None, prior=1.0)
        mcts_for_inspection = MCTS(model, num_simulations=25)
        mcts_for_inspection._expand(root)
        for _ in range(25):
            node = root
            while node.children:
                node = mcts_for_inspection._select_child(node)
            mcts_for_inspection._simulate_from(node)

        assert sum(c.visit_count for c in root.children.values()) == 25


class TestExpansionCallsPredictOnce:
    def test_predict_called_exactly_once_per_expansion(self):
        engine = make_engine(5, player_count=2)
        model = _FakeModel()
        mcts = MCTS(model, num_simulations=1)

        root = _MCTSNode(state=engine.get_state(), current_player=1, parent=None, prior=1.0)
        calls_before = len(model.calls)
        mcts._expand(root)

        assert len(model.calls) == calls_before + 1


class TestStructuralAlternation:
    def test_children_and_grandchildren_alternate_current_player(self):
        engine = make_engine(5, player_count=2)
        model = _FakeModel()
        mcts = MCTS(model, num_simulations=10)

        root = _MCTSNode(state=engine.get_state(), current_player=1, parent=None, prior=1.0)
        mcts._expand(root)
        for child in root.children.values():
            assert child.current_player != root.current_player
            mcts._expand(child)
            for grandchild in child.children.values():
                assert grandchild.current_player == root.current_player


class TestSelectChildAndBackupInIsolation:
    def _leaf(self, parent, prior, visit_count, value_sum, current_player=2):
        return _MCTSNode(
            state=parent.state,
            current_player=current_player,
            parent=parent,
            prior=prior,
            visit_count=visit_count,
            value_sum=value_sum,
        )

    def test_select_child_prefers_lower_child_value_all_else_equal(self):
        # child.value means "P(child's own mover wins)" — from the
        # SELECTING node's perspective, a low child value is good (the
        # opponent is unlikely to win), so it should be preferred when
        # priors/visit counts are otherwise tied.
        root = _MCTSNode(
            state={"size": 5}, current_player=1, parent=None, prior=1.0  # type: ignore[typeddict-item]
        )
        # good: value = 0.1 (bad for the opponent, i.e. good for us)
        good = self._leaf(root, prior=0.5, visit_count=10, value_sum=1.0)
        # bad: value = 0.9 (good for the opponent, i.e. bad for us)
        bad = self._leaf(root, prior=0.5, visit_count=10, value_sum=9.0)
        root.children = {MoveAction(Direction.UP): good, MoveAction(Direction.DOWN): bad}

        mcts = MCTS(_FakeModel(), num_simulations=1)
        selected = mcts._select_child(root)

        assert selected is good

    def test_backup_negates_value_for_the_parent(self):
        root = _MCTSNode(state={"size": 5}, current_player=1, parent=None, prior=1.0)  # type: ignore[typeddict-item]
        child = self._leaf(root, prior=1.0, visit_count=0, value_sum=0.0, current_player=2)
        root.children = {MoveAction(Direction.UP): child}

        mcts = MCTS(_FakeModel(), num_simulations=1)
        mcts._backup(child, value_player=2, value=0.0)  # player 2 has 0% win chance

        assert child.visit_count == 1
        assert child.value == pytest.approx(0.0)  # P(player 2 wins) = 0, as given
        assert root.visit_count == 1
        assert root.value == pytest.approx(1.0)  # P(player 1 wins) = 1 - 0
