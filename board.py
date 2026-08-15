from typing import List, Set, Tuple, Dict


size_to_walls_map: Dict[int, int] = {
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
        self.p1_pos: List[int] = [0, size // 2]
        self.p2_pos: List[int] = [size - 1, size // 2]

        # Wall Slots:
        # h_walls: (row, col) wall placed directly below cell (row, col)
        # v_walls: (row, col) wall placed directly to the right of cell (row, col)
        self.h_walls: Set[Tuple[int, int]] = set()
        self.v_walls: Set[Tuple[int, int]] = set()

        # Wall inventories
        self.p1_walls_left: int = size_to_walls(size)
        self.p2_walls_left: int = size_to_walls(size)

    def to_dict(self):
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
    def from_dict(cls, data):
        data_copy = data.copy()

        instance = cls(data_copy.at("size"))
        for key, value in data_copy.items():
            setattr(instance, key, value)
        return instance

    def get_state(self) -> dict:
        return self.to_dict()


if __name__ == "__main__":
    print("Hello")

    game = QuoridorBoard(5)

    print(game.to_dict())

