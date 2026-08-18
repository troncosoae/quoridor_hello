from abc import ABC, abstractmethod

from quoridor import pathfinding
from quoridor.actions import Action, MoveAction, WallAction
from quoridor.board import goal_cells
from quoridor.engine import Direction, EngineLike, WallOrientation
from quoridor.rendering import Renderer
from quoridor.timeouts import run_with_timeout

_DIRECTION_BY_DELTA: dict[tuple[int, int], Direction] = {
    (-1, 0): Direction.UP,
    (1, 0): Direction.DOWN,
    (0, -1): Direction.LEFT,
    (0, 1): Direction.RIGHT,
}

_WALL_ORIENTATION_TOKENS = {"h": WallOrientation.HORIZONTAL, "v": WallOrientation.VERTICAL}


class Agent(ABC):
    """The thing that decides and takes actions on behalf of one player.

    Like EngineLike, an Agent must only observe game state through the
    EngineLike interface passed into choose_action — never reach into a
    QuoridorEngine's `.board` directly, or it'll break the moment it's
    driving a RemoteEngine instead.
    """

    def __init__(self, player: int, renderer: Renderer | None = None):
        self.player = player
        self.renderer = renderer

    @abstractmethod
    def choose_action(self, engine: EngineLike) -> Action: ...


class CLIAgent(Agent):
    """Human player driven by terminal input."""

    def __init__(self, player: int, renderer: Renderer):
        super().__init__(player, renderer)

    def choose_action(self, engine: EngineLike) -> Action:
        assert self.renderer is not None
        print(self.renderer.render(engine.get_state()))

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


class BFSAgent(Agent):
    """Rule-based AI: always steps along the shortest unobstructed path to
    its goal row. Never places walls (deliberately shallow — no wall
    strategy, no lookahead beyond one BFS)."""

    # Bounds the whole decision (pathfinding + validity fallback checks),
    # not just the BFS call — defensive, see TimeoutExceededError's docstring.
    DECISION_TIMEOUT_SECONDS = 2.0

    def choose_action(self, engine: EngineLike) -> Action:
        if self.renderer is not None:
            print(self.renderer.render(engine.get_state()))

        def decide() -> Action:
            state = engine.get_state()
            own_pos = state["positions"][self.player - 1]
            goal = goal_cells(self.player, state["size"])

            path = pathfinding.bfs_shortest_path(
                set(state["h_walls"]), set(state["v_walls"]), state["size"],
                (own_pos[0], own_pos[1]), goal,
            )

            preferred: Direction | None = None
            if path is not None and len(path) >= 2:
                delta = (path[1][0] - own_pos[0], path[1][1] - own_pos[1])
                preferred = _DIRECTION_BY_DELTA[delta]

            if preferred is not None and engine.is_valid_move(self.player, preferred):
                return MoveAction(preferred)

            for direction in Direction:
                if direction != preferred and engine.is_valid_move(self.player, direction):
                    return MoveAction(direction)

            raise RuntimeError(f"BFSAgent for player {self.player} has no legal move")

        return run_with_timeout(decide, self.DECISION_TIMEOUT_SECONDS)
