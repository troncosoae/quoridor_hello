from abc import ABC, abstractmethod

from quoridor import pathfinding
from quoridor.actions import Action, MoveAction, WallAction
from quoridor.board import goal_cells
from quoridor.engine import Direction, EngineLike, WallOrientation
from quoridor.pathfinding import is_wall_between
from quoridor.rendering import CLIRenderer, Renderer
from quoridor.timeouts import run_with_timeout

_DIRECTION_BY_DELTA: dict[tuple[int, int], Direction] = {
    (-1, 0): Direction.UP,
    (1, 0): Direction.DOWN,
    (0, -1): Direction.LEFT,
    (0, 1): Direction.RIGHT,
}

_WALL_ORIENTATION_TOKENS = {"h": WallOrientation.HORIZONTAL, "v": WallOrientation.VERTICAL}


class UnsupportedGameSettingError(Exception):
    """An Agent was used in a game configuration it doesn't support."""


class Agent(ABC):
    """The thing that decides and takes actions on behalf of one player.

    Like EngineLike, an Agent must only observe game state through the
    EngineLike interface passed into choose_action — never reach into a
    QuoridorEngine's `.board` directly, or it'll break the moment it's
    driving a RemoteEngine instead.

    Enforcement of SUPPORTED_PLAYER_COUNTS happens per-entry-point (inside
    choose_action), not at construction time — direct construction of a
    restricted subclass doesn't itself raise. If a future method is added
    to Agent that reads engine state, it needs its own ensure_supports()
    call; GameRunner is currently the only real driver and it exclusively
    calls choose_action, so there's no live bypass today.
    """

    SUPPORTED_PLAYER_COUNTS: frozenset[int] | None = None  # None = supports any
    DECISION_TIMEOUT_SECONDS: float = 2.0

    def __init__(self, player: int, renderer: Renderer | None = None):
        self.player = player
        self.renderer = renderer

    @classmethod
    def supports(cls, player_count: int) -> bool:
        return cls.SUPPORTED_PLAYER_COUNTS is None or player_count in cls.SUPPORTED_PLAYER_COUNTS

    @classmethod
    def ensure_supports(cls, player_count: int) -> None:
        if not cls.supports(player_count):
            supported = (
                sorted(cls.SUPPORTED_PLAYER_COUNTS) if cls.SUPPORTED_PLAYER_COUNTS else "any"
            )
            raise UnsupportedGameSettingError(
                f"{cls.__name__} does not support {player_count}-player games "
                f"(supports: {supported})"
            )

    @abstractmethod
    def choose_action(self, engine: EngineLike) -> Action: ...


class CLIAgent(Agent):
    """Human player driven by terminal input. Works for any player count or
    board size — it just renders whatever state it's given and prompts."""

    SUPPORTED_PLAYER_COUNTS = None

    def __init__(self, player: int, renderer: Renderer):
        super().__init__(player, renderer)

    def choose_action(self, engine: EngineLike) -> Action:
        assert self.renderer is not None
        state = engine.get_state()
        self.ensure_supports(state["player_count"])
        print(self.renderer.render(state))

        while True:
            raw = input(
                f"Player {self.player} action "
                f"(e.g. 'move up', 'wall h 2 3'): "
            ).strip().lower()
            action = self._parse(raw)
            if action is None:
                print("Could not parse that command, try again.")
                continue
            if not self._is_legal(engine, action):
                print("That move isn't legal right now, try again.")
                continue
            return action

    def _is_legal(self, engine: EngineLike, action: Action) -> bool:
        if isinstance(action, MoveAction):
            return engine.is_valid_move(self.player, action.direction)
        return engine.is_valid_wall_placement(
            self.player, action.orientation, action.row, action.col
        )

    def _parse(self, raw: str) -> Action | None:
        parts = raw.split()

        if len(parts) == 2 and parts[0] == "move":
            try:
                return MoveAction(Direction(parts[1]))
            except ValueError:
                return None

        if len(parts) == 4 and parts[0] == "wall":
            orientation_token, row_str, col_str = parts[1], parts[2], parts[3]
            if orientation_token not in _WALL_ORIENTATION_TOKENS:
                return None
            if not row_str.isdigit() or not col_str.isdigit():
                return None
            return WallAction(
                _WALL_ORIENTATION_TOKENS[orientation_token], int(row_str), int(col_str)
            )

        return None


def _preferred_direction(
    own_pos: list[int],
    field: dict[tuple[int, int], int],
    h_walls: set[tuple[int, int]],
    v_walls: set[tuple[int, int]],
) -> Direction | None:
    """The direction to step to move one cell closer to goal, per a
    distance_field. A distance-value match alone isn't enough to pick a
    neighbor — a neighbor can reach the same reduced distance via a
    completely different route while the direct edge to it is
    wall-blocked, so the edge itself must also be confirmed open."""
    own_dist = field.get((own_pos[0], own_pos[1]))
    if own_dist is None or own_dist == 0:
        return None

    for delta, direction in _DIRECTION_BY_DELTA.items():
        neighbor = (own_pos[0] + delta[0], own_pos[1] + delta[1])
        if field.get(neighbor) == own_dist - 1 and not is_wall_between(
            h_walls, v_walls, own_pos[0], own_pos[1], neighbor[0], neighbor[1]
        ):
            return direction

    return None


def _shortest_path_choose_action(agent: Agent, engine: EngineLike) -> Action:
    # Fetched once and reused below (render + support-check + decision) —
    # deliberately outside run_with_timeout, matching how the old
    # render-only get_state() call was already excluded from the timeout.
    state = engine.get_state()
    agent.ensure_supports(state["player_count"])
    if agent.renderer is not None:
        print(agent.renderer.render(state))

    def decide() -> Action:
        h_walls, v_walls = set(state["h_walls"]), set(state["v_walls"])
        own_pos = state["positions"][agent.player - 1]
        goal = goal_cells(agent.player, state["size"])
        field = pathfinding.distance_field(h_walls, v_walls, state["size"], goal)
        preferred = _preferred_direction(own_pos, field, h_walls, v_walls)

        if preferred is not None and engine.is_valid_move(agent.player, preferred):
            return MoveAction(preferred)

        for direction in Direction:
            if direction != preferred and engine.is_valid_move(agent.player, direction):
                return MoveAction(direction)

        raise RuntimeError(f"No legal move for player {agent.player}")

    return run_with_timeout(decide, agent.DECISION_TIMEOUT_SECONDS)


class TwoPlayerBFSAgent(Agent):
    """Rule-based AI for 2-player games: always steps along the shortest
    unobstructed path to its goal (via a goal-outward distance field, not a
    per-turn point-to-point search). Never places walls (deliberately
    shallow — no wall strategy, no lookahead)."""

    SUPPORTED_PLAYER_COUNTS = frozenset({2})
    DECISION_TIMEOUT_SECONDS = 2.0

    def choose_action(self, engine: EngineLike) -> Action:
        return _shortest_path_choose_action(self, engine)


class FourPlayerBFSAgent(Agent):
    """Same shortest-path-via-distance-field logic as TwoPlayerBFSAgent,
    scoped to 4-player games. A separate class rather than one agent
    handling both counts, so each declares exactly what it supports."""

    SUPPORTED_PLAYER_COUNTS = frozenset({4})
    DECISION_TIMEOUT_SECONDS = 2.0

    def choose_action(self, engine: EngineLike) -> Action:
        return _shortest_path_choose_action(self, engine)


def build_agent(
    player: int, kind: str, player_count: int, renderer: Renderer | None = None
) -> Agent:
    if kind == "human":
        agent_cls: type[Agent] = CLIAgent
    elif player_count == 2:
        agent_cls = TwoPlayerBFSAgent
    elif player_count == 4:
        agent_cls = FourPlayerBFSAgent
    else:
        raise UnsupportedGameSettingError(f"No BFS agent available for {player_count}-player games")

    agent_cls.ensure_supports(player_count)
    return agent_cls(player, renderer if renderer is not None else CLIRenderer())
