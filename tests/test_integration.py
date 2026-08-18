import pytest

from quoridor.agents import Agent, FourPlayerBFSAgent, TwoPlayerBFSAgent
from quoridor.client import RemoteEngine, SeatTakenError
from quoridor.engine import Direction, InvalidMoveError
from quoridor.runner import GameRunner


class TestRemoteGameplay:
    def test_bfs_vs_bfs_full_game_over_http(self, live_server):
        base_url, _ = live_server
        remote = RemoteEngine(base_url)

        # Proves Agent/GameRunner code is unchanged whether the engine is
        # local or remote: this is the exact same GameRunner used against a
        # local QuoridorEngine in test_runner.py, here driving two BFS
        # agents to a real winner purely through HTTP round-trips.
        agents: dict[int, Agent] = {1: TwoPlayerBFSAgent(1), 2: TwoPlayerBFSAgent(2)}
        winner = GameRunner(remote, agents).run()

        assert winner in (1, 2)
        assert remote.winner() == winner

    def test_illegal_move_raises_invalid_move_error_locally(self, live_server):
        base_url, _ = live_server
        remote = RemoteEngine(base_url)

        with pytest.raises(InvalidMoveError):
            remote.move(2, Direction.UP)  # it's player 1's turn first

    def test_bfs_four_way_full_game_over_http(self, live_server_4p):
        base_url, _ = live_server_4p
        remote = RemoteEngine(base_url)

        agents: dict[int, Agent] = {p: FourPlayerBFSAgent(p) for p in (1, 2, 3, 4)}
        winner = GameRunner(remote, agents).run()

        assert winner in (1, 2, 3, 4)
        assert remote.winner() == winner


class TestSeatClaiming:
    def test_two_clients_cannot_both_claim_player_one(self, live_server):
        base_url, _ = live_server
        first_client = RemoteEngine(base_url)
        second_client = RemoteEngine(base_url)

        first_client.claim_player(1)

        with pytest.raises(SeatTakenError):
            second_client.claim_player(1)

    def test_two_clients_can_take_different_seats(self, live_server):
        base_url, _ = live_server
        first_client = RemoteEngine(base_url)
        second_client = RemoteEngine(base_url)

        first_client.claim_player(1)
        second_client.claim_player(2)  # should not raise
