from typing import TypedDict


class BoardState(TypedDict):
    size: int
    p1_pos: list[int]
    p2_pos: list[int]
    h_walls: list[tuple[int, int]]
    v_walls: list[tuple[int, int]]
    p1_walls_left: int
    p2_walls_left: int


size_to_walls_map: dict[int, int] = {
    5: 3,
    7: 7,
    9: 10
}


def size_to_walls(size: int) -> int:
    if size not in size_to_walls_map:
        raise ValueError("size is invalid")
    wall_count = size_to_walls_map.get(size)
    if not wall_count:
        raise RuntimeError("wall count should evaluate to true")
    return wall_count


class QuoridorBoard:
    def __init__(self, size: int = 9):
        if size not in size_to_walls_map:
            raise ValueError("size is invalid")
        self.size: int = size

        # Player Pawn Positions: [row, col]
        self.p1_pos: list[int] = [0, size // 2]
        self.p2_pos: list[int] = [size - 1, size // 2]

        # Wall Slots:
        # h_walls: (row, col) wall placed directly below cell (row, col)
        # v_walls: (row, col) wall placed directly to the right of cell (row, col)
        self.h_walls: set[tuple[int, int]] = set()
        self.v_walls: set[tuple[int, int]] = set()

        # Wall inventories
        self.p1_walls_left: int = size_to_walls(size)
        self.p2_walls_left: int = size_to_walls(size)

    def to_dict(self) -> BoardState:
        return {
            "size": self.size,
            "p1_pos": self.p1_pos,
            "p2_pos": self.p2_pos,
            "h_walls": list(self.h_walls),
            "v_walls": list(self.v_walls),
            "p1_walls_left": self.p1_walls_left,
            "p2_walls_left": self.p2_walls_left
        }

    @classmethod
    def from_dict(cls, data: BoardState) -> "QuoridorBoard":
        instance = cls(data["size"])
        instance.p1_pos = list(data["p1_pos"])
        instance.p2_pos = list(data["p2_pos"])
        instance.h_walls = set(data["h_walls"])
        instance.v_walls = set(data["v_walls"])
        instance.p1_walls_left = data["p1_walls_left"]
        instance.p2_walls_left = data["p2_walls_left"]
        return instance

    def get_state(self) -> BoardState:
        return self.to_dict()

class CLIRenderer:
    ROW_LABEL_WIDTH = 3

    @staticmethod
    def render(board: QuoridorBoard) -> str:
        state = board.get_state()
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

        label_pad = " " * CLIRenderer.ROW_LABEL_WIDTH
        header = label_pad + "".join(
            str(grid_c // 2) if grid_c % 2 == 0 else " " for grid_c in range(grid_cols)
        )
        lines = [header]

        for grid_r in range(grid_rows):
            if grid_r % 2 == 0:
                label = str(grid_r // 2).rjust(CLIRenderer.ROW_LABEL_WIDTH - 1) + " "
            else:
                label = label_pad
            row_str = "".join(grid[grid_r][grid_c] for grid_c in range(grid_cols))
            lines.append(label + row_str)

        lines.append(
            f"P1 walls left: {state['p1_walls_left']}  "
            f"P2 walls left: {state['p2_walls_left']}"
        )

        rendered = "\n".join(lines)
        print(rendered)
        return rendered


if __name__ == "__main__":
    game = QuoridorBoard(9)
    game.h_walls.add((2, 3))
    game.v_walls.add((4, 4))

    CLIRenderer.render(game)

