from dataclasses import dataclass
from typing import Any

from quoridor.board import BoardState
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


def legal_actions(engine: EngineLike, player: int, state: BoardState) -> list[Action]:
    """Every legal action for `player` in `state`. Pure — direct
    is_valid_move/is_valid_wall_placement enumeration, no torch involved
    (unlike quoridor.rl.encoding's legal_action_mask, which builds a
    tensor for network masking). Costs an O(size^2) sweep over wall slots,
    same as that mask — only call where that cost is actually wanted."""
    actions: list[Action] = [
        MoveAction(direction) for direction in Direction if engine.is_valid_move(player, direction)
    ]
    max_index = state["size"] - 2
    for orientation in WallOrientation:
        for row in range(max_index + 1):
            for col in range(max_index + 1):
                if engine.is_valid_wall_placement(player, orientation, row, col):
                    actions.append(WallAction(orientation, row, col))
    return actions


def action_to_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, MoveAction):
        return {"kind": "move", "direction": action.direction.value}
    return {
        "kind": "wall",
        "orientation": action.orientation.value,
        "row": action.row,
        "col": action.col,
    }


def action_from_dict(data: dict[str, Any]) -> Action:
    if data["kind"] == "move":
        return MoveAction(Direction(data["direction"]))
    if data["kind"] == "wall":
        return WallAction(WallOrientation(data["orientation"]), data["row"], data["col"])
    raise ValueError(f"unknown action kind: {data['kind']!r}")
