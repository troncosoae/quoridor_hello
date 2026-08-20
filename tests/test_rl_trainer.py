import math

from quoridor.game_store import FileGameStore, InMemoryGameStore
from quoridor.rl.cnn_model import CNNModel
from quoridor.rl.trainer import Trainer, TrainingConfig


def _tiny_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=2,
        lookback_batches=2,
        num_batches=2,
        max_plies_per_game=40,
        seed=0,
    )


def test_tiny_training_run_completes_with_in_memory_store(capsys):
    model = CNNModel(size=5, player_count=2)
    store = InMemoryGameStore()
    trainer = Trainer(model, store, _tiny_config())

    trainer.run()

    out = capsys.readouterr().out
    assert "cutoff" in out
    assert "loss=" in out
    for line in out.splitlines():
        if line.startswith("batch") and "loss=" in line:
            loss_str = line.split("loss=")[1].split(",")[0]
            assert not math.isnan(float(loss_str))


def test_tiny_training_run_with_file_store_produces_plain_text(tmp_path):
    path = tmp_path / "games.jsonl"
    model = CNNModel(size=5, player_count=2)
    store = FileGameStore(str(path))
    trainer = Trainer(model, store, _tiny_config())

    trainer.run()

    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == _tiny_config().batch_size * _tiny_config().num_batches
    assert all(line.startswith("{") for line in lines)
