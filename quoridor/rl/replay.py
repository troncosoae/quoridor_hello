from quoridor.game_store import GameRecord
from quoridor.rendering import Renderer


def replay_game(record: GameRecord, renderer: Renderer) -> None:
    """Step through a stored GameRecord move by move, printing the board
    before each ply and what was played. Kept deliberately minimal — the
    point is to be able to see what a training run actually produced, not
    to build a full replay UI."""
    for ply in record.plies:
        print(renderer.render(ply.state))
        print(f"Player {ply.current_player} ({ply.actor}) plays {ply.action}")
    if record.winner is not None:
        print(f"Winner: player {record.winner}")
    else:
        print("No winner (hit the ply cutoff)")
