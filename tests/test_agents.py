import time

import pytest

from quoridor import pathfinding
from quoridor.actions import MoveAction, WallAction
from quoridor.agents import BFSAgent, CLIAgent
from quoridor.board import QuoridorBoard
from quoridor.engine import Direction, QuoridorEngine, WallOrientation
from quoridor.rendering import CLIRenderer
from quoridor.timeouts import TimeoutExceededError


def make_engine(size=5):
    return QuoridorEngine(QuoridorBoard(size))


class TestCLIAgentParser:
    def test_parses_move(self):
        agent = CLIAgent(1, CLIRenderer())
        assert agent._parse("move up") == MoveAction(Direction.UP)

    def test_parses_horizontal_wall(self):
        agent = CLIAgent(1, CLIRenderer())
        assert agent._parse("wall h 2 3") == WallAction(WallOrientation.HORIZONTAL, 2, 3)

    def test_parses_vertical_wall(self):
        agent = CLIAgent(1, CLIRenderer())
        assert agent._parse("wall v 0 0") == WallAction(WallOrientation.VERTICAL, 0, 0)

    @pytest.mark.parametrize(
        "raw",
        ["", "move", "move sideways", "wall x 0 0", "wall h a 0", "jump up"],
    )
    def test_rejects_unparseable_input(self, raw):
        agent = CLIAgent(1, CLIRenderer())
        assert agent._parse(raw) is None


class TestCLIAgentChooseAction:
    def test_reprompts_on_bad_input_then_returns_good_action(self, monkeypatch, capsys):
        engine = make_engine(5)
        agent = CLIAgent(1, CLIRenderer())

        responses = iter(["not a command", "move down"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))

        action = agent.choose_action(engine)

        assert action == MoveAction(Direction.DOWN)
        assert "Could not parse" in capsys.readouterr().out

    def test_reprompts_on_illegal_move(self, monkeypatch, capsys):
        engine = make_engine(5)
        agent = CLIAgent(1, CLIRenderer())

        # UP is off the board for player 1 at the start; DOWN is legal.
        responses = iter(["move up", "move down"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))

        action = agent.choose_action(engine)

        assert action == MoveAction(Direction.DOWN)
        assert "isn't legal" in capsys.readouterr().out


class TestBFSAgent:
    def test_moves_toward_goal_on_empty_board(self):
        engine = make_engine(5)
        agent = BFSAgent(1)
        action = agent.choose_action(engine)
        assert action == MoveAction(Direction.DOWN)

    def test_player_two_moves_toward_its_own_goal(self):
        engine = make_engine(5)
        agent = BFSAgent(2)
        action = agent.choose_action(engine)
        assert action == MoveAction(Direction.UP)

    def test_decision_timeout_raises(self, monkeypatch):
        def slow_bfs(h_walls, v_walls, size, start, goal_row):
            time.sleep(0.2)
            return pathfinding.bfs_shortest_path(h_walls, v_walls, size, start, goal_row)

        monkeypatch.setattr(pathfinding, "bfs_shortest_path", slow_bfs)

        engine = make_engine(5)
        agent = BFSAgent(1)
        agent.DECISION_TIMEOUT_SECONDS = 0.01

        with pytest.raises(TimeoutExceededError):
            agent.choose_action(engine)

    def test_falls_back_when_preferred_step_is_onto_opponent(self):
        engine = make_engine(5)
        engine.board.p2_pos = [1, 2]  # directly below p1's start — p1's preferred step

        agent = BFSAgent(1)
        action = agent.choose_action(engine)

        assert isinstance(action, MoveAction)
        assert action.direction != Direction.DOWN
        assert engine.is_valid_move(1, action.direction)
