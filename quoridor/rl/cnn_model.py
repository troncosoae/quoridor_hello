import torch

from quoridor.board import BoardState
from quoridor.engine import EngineLike
from quoridor.rl.encoding import (
    action_size,
    encode_state,
    index_to_action,
    legal_action_mask,
    num_planes,
)
from quoridor.rl.model import Model, ModelPrediction
from quoridor.rl.network import QuoridorNet


class CNNModel(Model):
    """Thin adapter wrapping the existing QuoridorNet + encoding utilities
    behind the Model interface. One instance is tied to exactly one
    (size, player_count) — a CNN's conv/FC layer dimensions are fixed at
    construction, so SUPPORTED_BOARD_SIZES/SUPPORTED_PLAYER_COUNTS are
    always singletons here, never a broader set."""

    def __init__(self, size: int, player_count: int):
        self.size = size
        self.player_count = player_count
        self.SUPPORTED_PLAYER_COUNTS = frozenset({player_count})
        self.SUPPORTED_BOARD_SIZES = frozenset({size})

        self.network = QuoridorNet(
            size=size,
            player_count=player_count,
            in_channels=num_planes(player_count),
            action_size=action_size(size),
        )
        self.network.eval()

    def predict(self, engine: EngineLike, player: int, state: BoardState) -> ModelPrediction:
        x = encode_state(state, player).unsqueeze(0)
        mask = legal_action_mask(engine, player, state["size"]).unsqueeze(0)

        with torch.no_grad():
            policy_probs, win_probs = self.network.predict(x, mask)

        policy = {
            index_to_action(i, state["size"]): float(p)
            for i, p in enumerate(policy_probs[0])
            if mask[0, i]
        }
        return ModelPrediction(policy=policy, value=win_probs[0].tolist())
