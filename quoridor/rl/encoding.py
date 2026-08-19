import torch

from quoridor.actions import Action, MoveAction, WallAction
from quoridor.board import BoardState, size_to_walls
from quoridor.engine import Direction, EngineLike, WallOrientation

MOVE_DIRECTIONS: tuple[Direction, ...] = (
    Direction.UP,
    Direction.DOWN,
    Direction.LEFT,
    Direction.RIGHT,
)
WALL_ORIENTATIONS: tuple[WallOrientation, ...] = (
    WallOrientation.HORIZONTAL,
    WallOrientation.VERTICAL,
)


def num_planes(player_count: int) -> int:
    # per player: pawn position, walls remaining, to-move indicator; plus
    # one shared plane each for h_walls and v_walls.
    return 3 * player_count + 2


def action_size(size: int) -> int:
    wall_slots = (size - 1) ** 2
    return len(MOVE_DIRECTIONS) + len(WALL_ORIENTATIONS) * wall_slots


def encode_state(state: BoardState, current_player: int) -> torch.Tensor:
    """(C, size, size) float tensor. Wall planes use the same size x size
    grid as position planes for a uniform trunk input, even though wall
    slots only ever populate the [0, size - 2] sub-grid."""
    size = state["size"]
    player_count = state["player_count"]
    max_walls = size_to_walls(size, player_count)
    planes = torch.zeros(num_planes(player_count), size, size, dtype=torch.float32)

    channel = 0
    for player_index in range(player_count):
        row, col = state["positions"][player_index]
        planes[channel, row, col] = 1.0
        channel += 1

    h_channel, v_channel = channel, channel + 1
    for row, col in state["h_walls"]:
        planes[h_channel, row, col] = 1.0
    for row, col in state["v_walls"]:
        planes[v_channel, row, col] = 1.0
    channel += 2

    for player_index in range(player_count):
        planes[channel, :, :] = state["walls_left"][player_index] / max_walls
        channel += 1

    planes[channel + current_player - 1, :, :] = 1.0

    return planes


def action_to_index(action: Action, size: int) -> int:
    if isinstance(action, MoveAction):
        return MOVE_DIRECTIONS.index(action.direction)

    wall_slots = (size - 1) ** 2
    orientation_offset = WALL_ORIENTATIONS.index(action.orientation) * wall_slots
    local = action.row * (size - 1) + action.col
    return len(MOVE_DIRECTIONS) + orientation_offset + local


def index_to_action(index: int, size: int) -> Action:
    if index < len(MOVE_DIRECTIONS):
        return MoveAction(MOVE_DIRECTIONS[index])

    wall_slots = (size - 1) ** 2
    wall_index = index - len(MOVE_DIRECTIONS)
    orientation = WALL_ORIENTATIONS[wall_index // wall_slots]
    row, col = divmod(wall_index % wall_slots, size - 1)
    return WallAction(orientation, row, col)


def legal_action_mask(engine: EngineLike, player: int, size: int) -> torch.Tensor:
    """Boolean mask over the full action space. Walks every wall slot
    through is_valid_wall_placement, which runs a connected-components
    pathfinding check per call — fine for driving a demo forward pass, but
    the naive O(size^2) sweep here is not what a training loop's inner MCTS
    loop should be calling per node; that wants an incrementally maintained
    legality set instead."""
    mask = torch.zeros(action_size(size), dtype=torch.bool)

    for i, direction in enumerate(MOVE_DIRECTIONS):
        mask[i] = engine.is_valid_move(player, direction)

    wall_slots = (size - 1) ** 2
    offset = len(MOVE_DIRECTIONS)
    for orientation_i, orientation in enumerate(WALL_ORIENTATIONS):
        for row in range(size - 1):
            for col in range(size - 1):
                local = row * (size - 1) + col
                mask[offset + orientation_i * wall_slots + local] = engine.is_valid_wall_placement(
                    player, orientation, row, col
                )

    return mask
