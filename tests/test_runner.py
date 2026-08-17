from collections.abc import Sequence

from quoridor.actions import Action, MoveAction
from quoridor.agents import Agent
from quoridor.board import QuoridorBoard
from quoridor.engine import Direction, EngineLike, QuoridorEngine
from quoridor.runner import GameRunner


class ScriptedAgent(Agent):
    def __init__(self, player: int, actions: Sequence[Action]):
        super().__init__(player)
        self._actions = iter(actions)

    def choose_action(self, engine: EngineLike) -> Action:
        return next(self._actions)


def test_game_runner_drives_scripted_game_to_completion():
    engine = QuoridorEngine(QuoridorBoard(5))

    # p1 walks straight down its own column to the goal row; p2 sidesteps
    # out of p1's column so neither run blocks the other along the way.
    p1_moves = [MoveAction(Direction.DOWN)] * 4
    p2_moves = [MoveAction(Direction.LEFT), MoveAction(Direction.RIGHT), MoveAction(Direction.LEFT)]

    agents: dict[int, Agent] = {
        1: ScriptedAgent(1, p1_moves),
        2: ScriptedAgent(2, p2_moves),
    }

    winner = GameRunner(engine, agents).run()

    assert winner == 1
    assert engine.board.p1_pos == [4, 2]
