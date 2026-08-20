import pytest

from quoridor.actions import MoveAction
from quoridor.board import BoardState
from quoridor.engine import Direction
from quoridor.game_store import FileGameStore, GamePly, GameRecord, InMemoryGameStore


def make_record(game_id="g1", batch_index=0, winner=1):
    state: BoardState = {
        "size": 5,
        "player_count": 2,
        "positions": [[0, 2], [4, 2]],
        "walls_left": [3, 3],
        "h_walls": [],
        "v_walls": [],
    }
    ply = GamePly(
        state=state,
        current_player=1,
        action=MoveAction(Direction.DOWN),
        actor="bfs",
        legal_actions=[MoveAction(Direction.DOWN)],
    )
    return GameRecord(
        game_id=game_id, batch_index=batch_index, size=5, player_count=2,
        plies=[ply], winner=winner,
    )


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryGameStore()
    return FileGameStore(str(tmp_path / "games.jsonl"))


def test_get_game_returns_none_when_missing(store):
    assert store.get_game("nope") is None


def test_save_and_get_game_round_trips(store):
    record = make_record()
    store.save_game(record)

    fetched = store.get_game("g1")

    assert fetched is not None
    assert fetched.game_id == "g1"
    assert fetched.winner == 1
    assert fetched.plies[0].action == MoveAction(Direction.DOWN)
    assert fetched.plies[0].actor == "bfs"
    assert fetched.plies[0].legal_actions == [MoveAction(Direction.DOWN)]
    assert fetched.plies[0].state["positions"] == [[0, 2], [4, 2]]


def test_games_in_window_filters_by_batch_index(store):
    for batch_index in range(5):
        store.save_game(make_record(game_id=f"g{batch_index}", batch_index=batch_index))

    window = store.games_in_window(lookback_batches=2, upto_batch=4)

    assert {g.game_id for g in window} == {"g3", "g4"}


def test_games_in_window_clamps_at_zero(store):
    store.save_game(make_record(game_id="g0", batch_index=0))

    window = store.games_in_window(lookback_batches=5, upto_batch=1)

    assert {g.game_id for g in window} == {"g0"}


def test_file_store_reads_empty_when_file_does_not_exist(tmp_path):
    store = FileGameStore(str(tmp_path / "missing.jsonl"))
    assert store.games_in_window(lookback_batches=5, upto_batch=0) == []


def test_file_store_is_plain_json_lines(tmp_path):
    path = tmp_path / "games.jsonl"
    store = FileGameStore(str(path))
    store.save_game(make_record())

    lines = path.read_text().strip().splitlines()

    assert len(lines) == 1
    assert '"game_id": "g1"' in lines[0]
