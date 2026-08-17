from dataclasses import dataclass

from quoridor.engine import Direction, EngineLike, WallOrientation


@dataclass(frozen=True)
class MoveAction:
    direction: Direction


@dataclass(frozen=True)
class WallAction:
    orientation: WallOrientation
    row: int
    col: int


Action = MoveAction | WallAction


def apply_action(engine: EngineLike, action: Action) -> None:
    player = engine.current_player
    if isinstance(action, MoveAction):
        engine.move(player, action.direction)
    else:
        engine.place_wall(player, action.orientation, action.row, action.col)
