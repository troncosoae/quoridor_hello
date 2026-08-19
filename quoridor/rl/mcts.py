import math
from dataclasses import dataclass, field

from quoridor.actions import Action, apply_action
from quoridor.board import BoardState, QuoridorBoard, goal_cells
from quoridor.engine import QuoridorEngine
from quoridor.rl.model import Model, ModelPrediction


def _winner(state: BoardState) -> int | None:
    """A state-only reimplementation of QuoridorEngine.winner() — needs no
    engine/board object, since MCTS checks terminal status purely from
    stored BoardState snapshots at every node."""
    for player, pos in enumerate(state["positions"], start=1):
        if tuple(pos) in goal_cells(player, state["size"]):
            return player
    return None


def _simulate_action(
    state: BoardState, current_player: int, action: Action
) -> tuple[BoardState, int]:
    # BoardState has no current_player field (that only lives on
    # QuoridorEngine, which always initializes it to 1) — so it must be
    # carried alongside the state explicitly and restored by hand after
    # reconstruction, or turn order silently resets.
    engine = QuoridorEngine(QuoridorBoard.from_dict(state))
    engine.current_player = current_player
    apply_action(engine, action)
    return engine.get_state(), engine.current_player


@dataclass
class _MCTSNode:
    state: BoardState
    current_player: int
    parent: "_MCTSNode | None"
    prior: float
    children: dict[Action, "_MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        """P(this node's own acting player wins)."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    """2-player only, by design: the alternating-sign backup below assumes
    strictly-alternating, zero-sum turns. 4-player MCTS has real open
    unsolved design questions (kingmaker dynamics, a vector-valued
    non-zero-sum outcome) and is deliberately out of scope here — see
    MCTSAgent.MCTS_SUPPORTED_PLAYER_COUNTS.

    Operates entirely on local, disposable QuoridorEngine/QuoridorBoard
    instances reconstructed from BoardState snapshots — never touches
    whatever engine (local or remote) the starting state came from, so a
    simulation can never mutate real game state.
    """

    C_PUCT = 1.5

    def __init__(self, model: Model, num_simulations: int = 20):
        self.model = model
        self.num_simulations = num_simulations

    def run(self, state: BoardState, current_player: int) -> Action:
        root = _MCTSNode(state=state, current_player=current_player, parent=None, prior=1.0)
        self._expand(root)
        if not root.children:
            raise RuntimeError(f"No legal move for player {current_player}")

        for _ in range(self.num_simulations):
            node = root
            while node.children:
                node = self._select_child(node)
            self._simulate_from(node)

        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def _select_child(self, node: _MCTSNode) -> _MCTSNode:
        total_visits = sum(c.visit_count for c in node.children.values())

        def puct(child: _MCTSNode) -> float:
            # (1 - child.value): child.value means P(the CHILD's acting
            # player wins), and child.current_player is always the opponent
            # relative to `node` (turns strictly alternate) — selecting from
            # `node` must maximize `node`'s own mover's chances, so the
            # child's value has to be negated here.
            exploration = (
                self.C_PUCT * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            )
            return (1.0 - child.value) + exploration

        return max(node.children.values(), key=puct)

    def _expand(self, node: _MCTSNode) -> ModelPrediction | None:
        if _winner(node.state) is not None:
            return None

        sim_engine = QuoridorEngine(QuoridorBoard.from_dict(node.state))
        sim_engine.current_player = node.current_player
        prediction = self.model.predict(sim_engine, node.current_player, node.state)

        for action, prior in prediction.policy.items():
            child_state, child_player = _simulate_action(node.state, node.current_player, action)
            node.children[action] = _MCTSNode(
                state=child_state, current_player=child_player, parent=node, prior=prior
            )

        return prediction

    def _simulate_from(self, node: _MCTSNode) -> None:
        winner = _winner(node.state)
        if winner is not None:
            value = 1.0 if winner == node.current_player else 0.0
        else:
            # Reused for both the children's priors (inside _expand) and
            # this leaf's value — a single predict() call per expansion.
            prediction = self._expand(node)
            assert prediction is not None  # _winner already checked above
            value = prediction.value[node.current_player - 1]

        self._backup(node, node.current_player, value)

    def _backup(self, node: _MCTSNode | None, value_player: int, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value if node.current_player == value_player else (1.0 - value)
            node = node.parent
