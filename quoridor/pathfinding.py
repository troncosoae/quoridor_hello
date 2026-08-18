from collections import deque

_NEIGHBOR_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def is_wall_between(
    h_walls: set[tuple[int, int]],
    v_walls: set[tuple[int, int]],
    row: int,
    col: int,
    new_row: int,
    new_col: int,
) -> bool:
    if new_row == row + 1:  # moving down
        return (row, col) in h_walls or (row, col - 1) in h_walls
    if new_row == row - 1:  # moving up
        return (new_row, col) in h_walls or (new_row, col - 1) in h_walls
    if new_col == col + 1:  # moving right
        return (row, col) in v_walls or (row - 1, col) in v_walls
    if new_col == col - 1:  # moving left
        return (row, new_col) in v_walls or (row - 1, new_col) in v_walls
    return False


def bfs_shortest_path(
    h_walls: set[tuple[int, int]],
    v_walls: set[tuple[int, int]],
    size: int,
    start: tuple[int, int],
    goal_cells: frozenset[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    visited = {start}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    queue = deque([start])

    while queue:
        row, col = queue.popleft()
        if (row, col) in goal_cells:
            path = [(row, col)]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        for d_row, d_col in _NEIGHBOR_DELTAS:
            new_row, new_col = row + d_row, col + d_col
            if not (0 <= new_row < size and 0 <= new_col < size):
                continue
            if is_wall_between(h_walls, v_walls, row, col, new_row, new_col):
                continue
            if (new_row, new_col) in visited:
                continue
            visited.add((new_row, new_col))
            came_from[(new_row, new_col)] = (row, col)
            queue.append((new_row, new_col))

    return None


def distance_field(
    h_walls: set[tuple[int, int]],
    v_walls: set[tuple[int, int]],
    size: int,
    goal_cells: frozenset[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Multi-source BFS seeded from every goal cell at once: shortest
    distance (in steps) from each reachable cell to the nearest goal cell,
    respecting walls. Goal cells map to 0. A cell fully walled off from
    every goal is simply absent from the returned mapping.

    Unlike a single point-to-point search, one flood fill here answers the
    distance for every cell on the board, not just one starting position —
    e.g. every player's distance to their own goal can be read off the same
    field instead of running a separate search per player.
    """
    distances: dict[tuple[int, int], int] = {cell: 0 for cell in goal_cells}
    queue = deque(distances.keys())

    while queue:
        row, col = queue.popleft()
        d = distances[(row, col)]
        for d_row, d_col in _NEIGHBOR_DELTAS:
            new_row, new_col = row + d_row, col + d_col
            if not (0 <= new_row < size and 0 <= new_col < size):
                continue
            if (new_row, new_col) in distances:
                continue
            if is_wall_between(h_walls, v_walls, row, col, new_row, new_col):
                continue
            distances[(new_row, new_col)] = d + 1
            queue.append((new_row, new_col))

    return distances


def connected_components(
    h_walls: set[tuple[int, int]], v_walls: set[tuple[int, int]], size: int
) -> list[set[tuple[int, int]]]:
    """Partition every board cell into regions, given the current walls.

    Every cell in a region can reach every other cell in that region (that's
    what "connected" means here) — so whether a region contains a goal-row
    cell answers the reachability question for any pawn standing in it,
    without re-deriving a path per pawn. One full-board flood fill computes
    the answer for all players at once, instead of one BFS per player.
    """
    visited: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []

    for start_row in range(size):
        for start_col in range(size):
            start = (start_row, start_col)
            if start in visited:
                continue

            component: set[tuple[int, int]] = set()
            queue = deque([start])
            visited.add(start)
            while queue:
                row, col = queue.popleft()
                component.add((row, col))
                for d_row, d_col in _NEIGHBOR_DELTAS:
                    new_row, new_col = row + d_row, col + d_col
                    if not (0 <= new_row < size and 0 <= new_col < size):
                        continue
                    if is_wall_between(h_walls, v_walls, row, col, new_row, new_col):
                        continue
                    if (new_row, new_col) in visited:
                        continue
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))

            components.append(component)

    return components


def region_containing(
    components: list[set[tuple[int, int]]], cell: tuple[int, int]
) -> set[tuple[int, int]]:
    for component in components:
        if cell in component:
            return component
    raise ValueError(f"{cell} is not in any region — every cell belongs to exactly one")
