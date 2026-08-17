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
