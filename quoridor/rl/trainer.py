import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from quoridor.agents import Agent, ModelAgent, TwoPlayerBFSAgent
from quoridor.board import QuoridorBoard
from quoridor.engine import QuoridorEngine
from quoridor.game_store import GameStore
from quoridor.rl.cnn_model import CNNModel
from quoridor.rl.encoding import action_size, action_to_index, encode_state
from quoridor.rl.targets import StateTarget, derive_training_targets
from quoridor.runner import GameRunner


@dataclass
class TrainingConfig:
    batch_size: int = 1000
    """Self-play games per batch."""
    lookback_batches: int = 5
    """How many of the most recent batches (including the one just played)
    to train on — a sliding window, not the whole history and not just the
    newest batch."""
    num_batches: int = 10
    exploration_temperature: float = 1.0
    """ModelAgent's self-play sampling knob — 0.0 is deterministic argmax,
    >0.0 samples from policy ** (1/temperature)."""
    policy_temperature: float = 1.0
    """A separate knob: sharpens/flattens the one-step-lookahead policy
    target during derivation, unrelated to exploration during play."""
    opponent_mix: dict[str, float] = field(
        default_factory=lambda: {"self": 0.5, "bfs": 0.5}
    )
    """Fraction of games played against each opponent kind. Naturally
    2-player-specific (a global split across exactly two seats) — the
    eventual 4-player generalization would need per-seat composition."""
    max_plies_per_game: int = 200
    """A cycle safety net, not a "typical game length" figure — nothing in
    QuoridorEngine/GameRunner stops a pawn oscillating forever, and a fully
    deterministic self-play match (temperature=0.0) is a pure function of
    state, so an exact repeat would cycle indefinitely without this."""
    learning_rate: float = 1e-3
    training_epochs_per_batch: int = 1
    checkpoint_dir: str | None = None
    seed: int | None = None


def _weighted_choice(rng: random.Random, mix: dict[str, float]) -> str:
    kinds = list(mix.keys())
    weights = list(mix.values())
    return rng.choices(kinds, weights=weights, k=1)[0]


class Trainer:
    """Naive first training loop: self-play generates data, Monte Carlo
    state-value + one-step-lookahead policy targets are derived from a
    sliding window of recent batches, and the same network is warm-started
    (never reinitialized) across every batch — matching AlphaZero/TD-Gammon
    practice. Deliberately simple: no MCTS-guided self-play, no parallel
    game generation. Scoped by the CNNModel passed in (5x5 2-player for
    this pass), but nothing in this class hardcodes that — the training
    logic reads player_count/size off the model, not off literals."""

    def __init__(self, model: CNNModel, store: GameStore, config: TrainingConfig):
        self.model = model
        self.store = store
        self.config = config
        self.optimizer = torch.optim.Adam(model.network.parameters(), lr=config.learning_rate)
        self._rng = random.Random(config.seed)

    def run(self) -> None:
        for batch_index in range(self.config.num_batches):
            cutoffs = self._play_batch(batch_index)
            print(
                f"batch {batch_index}: {cutoffs}/{self.config.batch_size} games "
                "hit the max_plies cutoff"
            )

            window = self.store.games_in_window(self.config.lookback_batches, batch_index)
            targets = derive_training_targets(window, self.config.policy_temperature)

            loss = float("nan")
            for _ in range(self.config.training_epochs_per_batch):
                loss = self._train_step(targets)
            print(f"batch {batch_index}: loss={loss:.4f}, {len(targets)} distinct states in window")

            self._checkpoint(batch_index)

    def _play_batch(self, batch_index: int) -> int:
        cutoffs = 0
        for game_index in range(self.config.batch_size):
            opponent_kind = _weighted_choice(self._rng, self.config.opponent_mix)
            model_seat = self._rng.choice([1, 2])
            opponent_seat = 2 if model_seat == 1 else 1

            engine = QuoridorEngine(QuoridorBoard(self.model.size, self.model.player_count))
            model_agent = ModelAgent(
                model_seat, self.model, temperature=self.config.exploration_temperature
            )
            opponent_agent: Agent
            if opponent_kind == "self":
                opponent_agent = ModelAgent(
                    opponent_seat, self.model, temperature=self.config.exploration_temperature
                )
            else:
                opponent_agent = TwoPlayerBFSAgent(opponent_seat)

            agents: dict[int, Agent] = {model_seat: model_agent, opponent_seat: opponent_agent}
            game_id = f"batch{batch_index}-game{game_index}"

            winner = GameRunner(
                engine, agents,
                store=self.store, game_id=game_id, batch_index=batch_index,
                max_plies=self.config.max_plies_per_game,
            ).run()
            if winner is None:
                cutoffs += 1
        return cutoffs

    def _train_step(self, targets: dict[str, StateTarget]) -> float:
        if not targets:
            return float("nan")

        self.model.network.train()
        self.optimizer.zero_grad()

        size = self.model.size
        n_actions = action_size(size)

        xs = []
        policy_targets = []
        value_targets = []
        masks = []
        for target in targets.values():
            xs.append(encode_state(target.state, target.current_player))

            mask = torch.zeros(n_actions, dtype=torch.bool)
            for action in target.legal_actions:
                mask[action_to_index(action, size)] = True
            masks.append(mask)

            policy_row = torch.zeros(n_actions, dtype=torch.float32)
            for action, weight in target.policy.items():
                policy_row[action_to_index(action, size)] = weight
            policy_targets.append(policy_row)

            value_targets.append(torch.tensor(target.value, dtype=torch.float32))

        x = torch.stack(xs)
        mask_t = torch.stack(masks)
        policy_target = torch.stack(policy_targets)
        value_target = torch.stack(value_targets)

        policy_logits, value_logits = self.model.network.forward(x)
        # A large finite penalty, not float("-inf"): policy_target is exactly
        # 0.0 on every illegal action, and 0.0 * (-inf) is nan, not 0 — that
        # nan would poison the whole row's loss (and every masked-target row
        # entirely) once multiplied through below. -1e9 masks just as
        # effectively for softmax purposes without that failure mode.
        policy_logits = policy_logits.masked_fill(~mask_t, -1e9)

        has_policy_target = policy_target.sum(dim=-1) > 0.0
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(policy_target * policy_log_probs).sum(dim=-1)
        policy_loss = (policy_loss * has_policy_target).sum() / has_policy_target.sum().clamp(min=1)

        value_log_probs = F.log_softmax(value_logits, dim=-1)
        value_loss = -(value_target * value_log_probs).sum(dim=-1).mean()

        loss = policy_loss + value_loss
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        self.model.network.eval()
        return float(loss.item())

    def _checkpoint(self, batch_index: int) -> None:
        if self.config.checkpoint_dir is None:
            return
        torch.save(
            self.model.network.state_dict(),
            f"{self.config.checkpoint_dir}/batch_{batch_index:04d}.pt",
        )
