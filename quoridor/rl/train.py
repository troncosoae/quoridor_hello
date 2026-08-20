import argparse

from quoridor.game_store import FileGameStore, GameStore, InMemoryGameStore
from quoridor.rl.cnn_model import CNNModel
from quoridor.rl.trainer import Trainer, TrainingConfig


def _build_store(kind: str, path: str | None) -> GameStore:
    if kind == "memory":
        return InMemoryGameStore()
    if path is None:
        raise ValueError("--store file requires --store-path")
    return FileGameStore(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN model via self-play.")
    parser.add_argument("--size", type=int, default=5, choices=[5, 7, 9])
    parser.add_argument("--players", type=int, default=2, choices=[2, 4])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--lookback-batches", type=int, default=5)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--exploration-temperature", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--max-plies-per-game", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--training-epochs-per-batch", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--store", choices=["memory", "file"], default="memory")
    parser.add_argument("--store-path", type=str, default=None)
    args = parser.parse_args()

    model = CNNModel(size=args.size, player_count=args.players)
    store = _build_store(args.store, args.store_path)
    config = TrainingConfig(
        batch_size=args.batch_size,
        lookback_batches=args.lookback_batches,
        num_batches=args.num_batches,
        exploration_temperature=args.exploration_temperature,
        policy_temperature=args.policy_temperature,
        max_plies_per_game=args.max_plies_per_game,
        learning_rate=args.learning_rate,
        training_epochs_per_batch=args.training_epochs_per_batch,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    Trainer(model, store, config).run()


if __name__ == "__main__":
    main()
