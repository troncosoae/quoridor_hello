import pytest

from quoridor.pathfinding import (
    bfs_shortest_path,
    connected_components,
    distance_field,
    is_wall_between,
    region_containing,
)


class TestIsWallBetween:
    def test_no_walls_no_block(self):
        assert is_wall_between(set(), set(), 0, 0, 1, 0) is False

    def test_horizontal_wall_blocks_directly_below(self):
        h_walls = {(0, 0)}
        assert is_wall_between(h_walls, set(), 0, 0, 1, 0) is True

    def test_horizontal_wall_blocks_both_columns_it_spans(self):
        h_walls = {(0, 0)}
        assert is_wall_between(h_walls, set(), 0, 1, 1, 1) is True

    def test_horizontal_wall_blocks_moving_up_too(self):
        h_walls = {(0, 0)}
        assert is_wall_between(h_walls, set(), 1, 0, 0, 0) is True

    def test_vertical_wall_blocks_moving_right(self):
        v_walls = {(0, 0)}
        assert is_wall_between(set(), v_walls, 0, 0, 0, 1) is True

    def test_vertical_wall_blocks_both_rows_it_spans(self):
        v_walls = {(0, 0)}
        assert is_wall_between(set(), v_walls, 1, 0, 1, 1) is True

    def test_non_adjacent_cells_never_blocked(self):
        assert is_wall_between({(0, 0)}, {(0, 0)}, 2, 2, 2, 3) is False


def _row_goal(row, size=5):
    return frozenset((row, c) for c in range(size))


class TestBfsShortestPath:
    def test_finds_direct_path_on_empty_board(self):
        path = bfs_shortest_path(set(), set(), 5, (0, 2), _row_goal(4))
        assert path is not None
        assert path[0] == (0, 2)
        assert path[-1][0] == 4

    def test_no_path_when_fully_boxed_in(self):
        # Wall directly below (0,0) spanning cols 0-1, and a wall to the
        # right of (0,0) spanning rows 0-1 traps a pawn started at (0,0)
        # in the corner (mirrors tests/test_engine.py's
        # test_wall_fully_blocking_path_rejected scenario, at the
        # pathfinding layer this time).
        h_walls = {(0, 0)}
        v_walls = {(0, 0)}
        path = bfs_shortest_path(h_walls, v_walls, 5, (0, 0), _row_goal(4))
        assert path is None

    def test_path_routes_around_a_wall(self):
        # Wall spans cols 0-1 directly below row 0; a pawn at (0, 0) can
        # still reach row 4 by going right first.
        h_walls = {(0, 0)}
        path = bfs_shortest_path(h_walls, set(), 5, (0, 0), _row_goal(4))
        assert path is not None
        assert path[-1][0] == 4

    def test_start_already_on_goal_row(self):
        path = bfs_shortest_path(set(), set(), 5, (4, 2), _row_goal(4))
        assert path == [(4, 2)]

    def test_goal_can_be_a_column_not_just_a_row(self):
        column_goal = frozenset((r, 4) for r in range(5))
        path = bfs_shortest_path(set(), set(), 5, (2, 0), column_goal)
        assert path is not None
        assert path[-1][1] == 4


class TestConnectedComponents:
    def test_empty_board_is_one_region(self):
        components = connected_components(set(), set(), 5)
        assert len(components) == 1
        assert len(components[0]) == 25

    def test_every_cell_appears_in_exactly_one_region(self):
        h_walls = {(0, 0), (2, 2)}
        v_walls = {(1, 1)}
        components = connected_components(h_walls, v_walls, 5)
        all_cells = {cell for region in components for cell in region}
        assert all_cells == {(r, c) for r in range(5) for c in range(5)}
        assert sum(len(region) for region in components) == 25

    def test_boxed_in_corner_is_its_own_singleton_region(self):
        # Same trap as TestBfsShortestPath.test_no_path_when_fully_boxed_in.
        h_walls = {(0, 0)}
        v_walls = {(0, 0)}
        components = connected_components(h_walls, v_walls, 5)
        corner_region = region_containing(components, (0, 0))
        assert corner_region == {(0, 0)}

    def test_region_reaching_goal_row(self):
        components = connected_components(set(), set(), 5)
        region = region_containing(components, (0, 2))
        assert any(row == 4 for row, _ in region)

    def test_boxed_in_region_does_not_reach_goal_row(self):
        h_walls = {(0, 0)}
        v_walls = {(0, 0)}
        components = connected_components(h_walls, v_walls, 5)
        region = region_containing(components, (0, 0))
        assert not any(row == 4 for row, _ in region)


class TestRegionContaining:
    def test_raises_for_a_cell_outside_the_board(self):
        components = connected_components(set(), set(), 5)
        with pytest.raises(ValueError, match=r"\(9, 9\)"):
            region_containing(components, (9, 9))


class TestDistanceField:
    def test_goal_cells_map_to_zero(self):
        goal = _row_goal(4)
        field = distance_field(set(), set(), 5, goal)
        for cell in goal:
            assert field[cell] == 0

    def test_straight_line_distances_on_empty_board(self):
        field = distance_field(set(), set(), 5, _row_goal(4))
        assert field[(0, 2)] == 4
        assert field[(3, 2)] == 1
        assert field[(4, 2)] == 0

    def test_wall_increases_the_routed_distance(self):
        # Wall spans cols 0-1 directly below row 0, forcing (0,0) to detour
        # right before it can head down (same setup as
        # TestBfsShortestPath.test_path_routes_around_a_wall).
        h_walls = {(0, 0)}
        field_without_wall = distance_field(set(), set(), 5, _row_goal(4))
        field_with_wall = distance_field(h_walls, set(), 5, _row_goal(4))
        assert field_with_wall[(0, 0)] > field_without_wall[(0, 0)]

    def test_unreachable_cell_is_absent(self):
        # Same corner trap as
        # TestConnectedComponents.test_boxed_in_corner_is_its_own_singleton_region.
        h_walls = {(0, 0)}
        v_walls = {(0, 0)}
        field = distance_field(h_walls, v_walls, 5, _row_goal(4))
        assert (0, 0) not in field
