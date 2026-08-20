from collections.abc import Sequence

from quoridor.actions import Action, MoveAction
from quoridor.agents import Agent
from quoridor.board import QuoridorBoard
from quoridor.engine import Direction, EngineLike, QuoridorEngine
from quoridor.game_store import InMemoryGameStore
from quoridor.runner import GameRunner


class ScriptedAgent(Agent):
    KIND = "scripted"

    def __init__(self, player: int, actions: Sequence[Action]):
        super().__init__(player)
        self._actions = iter(actions)

    def choose_action(self, engine: EngineLike) -> Action:
        return next(self._actions)


def _p1_p2_moves():
    # p1 walks straight down its own column to the goal row; p2 sidesteps
    # out of p1's column so neither run blocks the other along the way.
    p1_moves = [MoveAction(Direction.DOWN)] * 4
    p2_moves = [MoveAction(Direction.LEFT), MoveAction(Direction.RIGHT), MoveAction(Direction.LEFT)]
    return p1_moves, p2_moves


def test_game_runner_drives_scripted_game_to_completion():
    engine = QuoridorEngine(QuoridorBoard(5))
    p1_moves, p2_moves = _p1_p2_moves()

    agents: dict[int, Agent] = {
        1: ScriptedAgent(1, p1_moves),
        2: ScriptedAgent(2, p2_moves),
    }

    winner = GameRunner(engine, agents).run()

    assert winner == 1
    assert engine.board.positions[0] == [4, 2]


def test_game_runner_without_a_store_does_not_touch_recording_at_all():
    # store=None must remain byte-identical to pre-recording behavior — no
    # legal_actions sweep, no GameRecord ever constructed.
    engine = QuoridorEngine(QuoridorBoard(5))
    p1_moves, p2_moves = _p1_p2_moves()
    agents: dict[int, Agent] = {1: ScriptedAgent(1, p1_moves), 2: ScriptedAgent(2, p2_moves)}

    runner = GameRunner(engine, agents, store=None)
    winner = runner.run()

    assert winner == 1


def test_game_runner_records_every_ply_when_given_a_store():
    engine = QuoridorEngine(QuoridorBoard(5))
    p1_moves, p2_moves = _p1_p2_moves()
    agents: dict[int, Agent] = {1: ScriptedAgent(1, p1_moves), 2: ScriptedAgent(2, p2_moves)}
    store = InMemoryGameStore()

    winner = GameRunner(engine, agents, store=store, game_id="g1", batch_index=3).run()

    record = store.get_game("g1")
    assert record is not None
    assert record.batch_index == 3
    assert record.winner == winner == 1
    assert len(record.plies) == len(p1_moves) + len(p2_moves)
    assert record.plies[0].actor == "scripted"
    assert record.plies[0].current_player == 1
    assert record.plies[0].action == p1_moves[0]
    assert MoveAction(Direction.DOWN) in record.plies[0].legal_actions


def test_game_runner_stops_at_max_plies_with_no_winner():
    # Two BFS-like scripted agents that just bounce sideways forever —
    # never reach a goal, so the only way this loop ever terminates is the
    # max_plies cutoff.
    engine = QuoridorEngine(QuoridorBoard(5))
    bounce = [MoveAction(Direction.LEFT), MoveAction(Direction.RIGHT)] * 10
    agents: dict[int, Agent] = {1: ScriptedAgent(1, bounce), 2: ScriptedAgent(2, bounce)}
    store = InMemoryGameStore()

    winner = GameRunner(
        engine, agents, store=store, game_id="g2", batch_index=0, max_plies=5
    ).run()

    assert winner is None
    record = store.get_game("g2")
    assert record is not None
    assert record.winner is None
    assert len(record.plies) == 5
