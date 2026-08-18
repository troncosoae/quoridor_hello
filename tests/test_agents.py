import time

import pytest

from quoridor import pathfinding
from quoridor.actions import MoveAction, WallAction
from quoridor.agents import (
    CLIAgent,
    FourPlayerBFSAgent,
    TwoPlayerBFSAgent,
    UnsupportedGameSettingError,
    _preferred_direction,
    build_agent,
)
from quoridor.board import VALID_PLAYER_COUNTS, QuoridorBoard
from quoridor.engine import Direction, QuoridorEngine, WallOrientation
from quoridor.rendering import CLIRenderer
from quoridor.timeouts import TimeoutExceededError


def make_engine(size=5, player_count=2):
    return QuoridorEngine(QuoridorBoard(size, player_count))


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


class TestAgentSupportDeclarations:
    def test_cli_agent_supports_any_player_count(self):
        assert CLIAgent.supports(2) is True
        assert CLIAgent.supports(4) is True

    def test_two_player_bfs_agent_supports_only_two(self):
        assert TwoPlayerBFSAgent.supports(2) is True
        assert TwoPlayerBFSAgent.supports(4) is False

    def test_four_player_bfs_agent_supports_only_four(self):
        assert FourPlayerBFSAgent.supports(4) is True
        assert FourPlayerBFSAgent.supports(2) is False

    def test_ensure_supports_raises_for_unsupported_count(self):
        with pytest.raises(UnsupportedGameSettingError):
            TwoPlayerBFSAgent.ensure_supports(4)
        with pytest.raises(UnsupportedGameSettingError):
            FourPlayerBFSAgent.ensure_supports(2)

    def test_choose_action_enforces_support_at_runtime(self):
        # Not just the static classmethod — actually using the agent against
        # a mismatched engine must raise too.
        with pytest.raises(UnsupportedGameSettingError):
            TwoPlayerBFSAgent(1).choose_action(make_engine(5, player_count=4))
        with pytest.raises(UnsupportedGameSettingError):
            FourPlayerBFSAgent(1).choose_action(make_engine(5, player_count=2))

    def test_bfs_agent_support_covers_every_valid_player_count_exactly_once(self):
        # Regression lock: if a future player count is added to board.py
        # without a matching agent decision, this fails loudly instead of
        # only surfacing as a runtime error nobody thought to test.
        combined = (
            TwoPlayerBFSAgent.SUPPORTED_PLAYER_COUNTS | FourPlayerBFSAgent.SUPPORTED_PLAYER_COUNTS
        )
        assert combined == VALID_PLAYER_COUNTS
        assert TwoPlayerBFSAgent.SUPPORTED_PLAYER_COUNTS.isdisjoint(
            FourPlayerBFSAgent.SUPPORTED_PLAYER_COUNTS
        )


class TestBuildAgent:
    def test_human_kind_builds_cli_agent(self):
        agent = build_agent(1, "human", player_count=4)
        assert isinstance(agent, CLIAgent)

    def test_bfs_kind_builds_two_player_agent_for_two_player_games(self):
        agent = build_agent(1, "bfs", player_count=2)
        assert isinstance(agent, TwoPlayerBFSAgent)

    def test_bfs_kind_builds_four_player_agent_for_four_player_games(self):
        agent = build_agent(1, "bfs", player_count=4)
        assert isinstance(agent, FourPlayerBFSAgent)


class TestPreferredDirection:
    def test_returns_none_when_already_on_goal(self):
        field = {(4, 2): 0}
        assert _preferred_direction([4, 2], field, set(), set()) is None

    def test_returns_none_when_unreachable(self):
        assert _preferred_direction([0, 0], {}, set(), set()) is None

    def test_does_not_pick_a_wall_blocked_neighbor_with_a_coincidentally_matching_distance(self):
        # Exact counterexample found in design review: (0,0)'s true
        # distance is 6, reached only via (0,1). Its DOWN neighbor (1,0)
        # independently has distance 5 (== own_dist - 1) via a completely
        # different route, while the direct edge (0,0)-(1,0) is
        # wall-blocked. A naive "does the neighbor's value match" check
        # would wrongly pick DOWN.
        h_walls = {(0, 0), (2, 0)}
        v_walls: set[tuple[int, int]] = set()
        goal = frozenset((4, c) for c in range(5))
        field = pathfinding.distance_field(h_walls, v_walls, 5, goal)

        assert field[(0, 0)] == 6
        assert field[(1, 0)] == 5  # confirms the coincidental match exists

        direction = _preferred_direction([0, 0], field, h_walls, v_walls)

        assert direction != Direction.DOWN
        assert direction == Direction.RIGHT


class TestTwoPlayerBFSAgent:
    def test_moves_toward_goal_on_empty_board(self):
        engine = make_engine(5)
        agent = TwoPlayerBFSAgent(1)
        action = agent.choose_action(engine)
        assert action == MoveAction(Direction.DOWN)

    def test_player_two_moves_toward_its_own_goal(self):
        engine = make_engine(5)
        agent = TwoPlayerBFSAgent(2)
        action = agent.choose_action(engine)
        assert action == MoveAction(Direction.UP)

    def test_decision_timeout_raises(self, monkeypatch):
        def slow_distance_field(h_walls, v_walls, size, goal_cells):
            time.sleep(0.2)
            return pathfinding.distance_field(h_walls, v_walls, size, goal_cells)

        monkeypatch.setattr(pathfinding, "distance_field", slow_distance_field)

        engine = make_engine(5)
        agent = TwoPlayerBFSAgent(1)
        agent.DECISION_TIMEOUT_SECONDS = 0.01

        with pytest.raises(TimeoutExceededError):
            agent.choose_action(engine)

    def test_falls_back_when_preferred_step_is_onto_opponent(self):
        engine = make_engine(5)
        engine.board.positions[1] = [1, 2]  # directly below p1's start — p1's preferred step

        agent = TwoPlayerBFSAgent(1)
        action = agent.choose_action(engine)

        assert isinstance(action, MoveAction)
        assert action.direction != Direction.DOWN
        assert engine.is_valid_move(1, action.direction)


class TestFourPlayerBFSAgent:
    def test_player_three_moves_toward_rightmost_column(self):
        engine = make_engine(5, player_count=4)
        agent = FourPlayerBFSAgent(3)
        action = agent.choose_action(engine)
        assert action == MoveAction(Direction.RIGHT)

    def test_player_four_moves_toward_leftmost_column(self):
        engine = make_engine(5, player_count=4)
        agent = FourPlayerBFSAgent(4)
        action = agent.choose_action(engine)
        assert action == MoveAction(Direction.LEFT)
