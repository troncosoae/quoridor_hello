from abc import ABC, abstractmethod

from quoridor.board import BoardState


class Renderer(ABC):
    """Turns a BoardState into a human-facing view.

    Implementations must be pure: no printing, no I/O. The caller decides
    what to do with the returned representation (print it, send it to a
    remote client, feed it to a GUI widget, etc).
    """

    @abstractmethod
    def render(self, state: BoardState) -> str: ...


class CLIRenderer(Renderer):
    ROW_LABEL_WIDTH = 3

    def render(self, state: BoardState) -> str:
        size = state["size"]

        grid_rows = 2 * size - 1
        grid_cols = 2 * size - 1
        grid = [[" " for _ in range(grid_cols)] for _ in range(grid_rows)]

        for r in range(size):
            for c in range(size):
                grid[2 * r][2 * c] = "."

        p1_row, p1_col = state["p1_pos"]
        p2_row, p2_col = state["p2_pos"]
        grid[2 * p1_row][2 * p1_col] = "1"
        grid[2 * p2_row][2 * p2_col] = "2"

        for r, c in state["h_walls"]:
            grid[2 * r + 1][2 * c] = "-"
            grid[2 * r + 1][2 * c + 1] = "-"
            grid[2 * r + 1][2 * c + 2] = "-"

        for r, c in state["v_walls"]:
            grid[2 * r][2 * c + 1] = "|"
            grid[2 * r + 1][2 * c + 1] = "|"
            grid[2 * r + 2][2 * c + 1] = "|"

        label_pad = " " * self.ROW_LABEL_WIDTH
        header = label_pad + "".join(
            str(grid_c // 2) if grid_c % 2 == 0 else " " for grid_c in range(grid_cols)
        )
        lines = [header]

        for grid_r in range(grid_rows):
            if grid_r % 2 == 0:
                label = str(grid_r // 2).rjust(self.ROW_LABEL_WIDTH - 1) + " "
            else:
                label = label_pad
            row_str = "".join(grid[grid_r][grid_c] for grid_c in range(grid_cols))
            lines.append(label + row_str)

        lines.append(
            f"P1 walls left: {state['p1_walls_left']}  "
            f"P2 walls left: {state['p2_walls_left']}"
        )

        return "\n".join(lines)
