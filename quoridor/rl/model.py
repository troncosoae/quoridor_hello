from abc import ABC, abstractmethod
from dataclasses import dataclass

from quoridor.actions import Action
from quoridor.board import BoardState
from quoridor.engine import EngineLike


@dataclass
class ModelPrediction:
    policy: dict[Action, float]
    """Legal actions only — already masked and renormalized. A model must
    never return a nonzero weight for an illegal action."""

    value: list[float]
    """Length == the player_count this model was built for.
    value[i] = P(player i+1 wins)."""


class Model(ABC):
    """The swappable prediction backend an Agent (ModelAgent/MCTSAgent) sits
    on top of. Zero dependency on any particular network framework or
    architecture — a future non-CNN model implements the same interface and
    plugs into the same agents unchanged.

    Unlike Agent.SUPPORTED_PLAYER_COUNTS (which defaults to "any"), there is
    deliberately no default here: a model's compatibility is tied to
    concrete weights/architecture (e.g. a CNN's layer dimensions are fixed
    to one board size), so every concrete Model must declare both sets
    explicitly — "unrestricted" would almost always be a lie.
    """

    SUPPORTED_PLAYER_COUNTS: frozenset[int]
    SUPPORTED_BOARD_SIZES: frozenset[int]

    @abstractmethod
    def predict(self, engine: EngineLike, player: int, state: BoardState) -> ModelPrediction:
        """`engine` is used to determine legal actions (e.g. via
        is_valid_move/is_valid_wall_placement) for masking — implementations
        must never return a nonzero policy weight for an illegal action."""
        ...
