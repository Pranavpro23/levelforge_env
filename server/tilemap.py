from typing import Literal, List, Tuple

# 1. TILE TYPE
Tile = Literal[".", "#", "^", "$", "E", "P", "G"]
# "." = empty space
# "#" = wall or platform
# "^" = spike (kills player)
# "$" = coin (collectible)
# "E" = enemy
# "P" = player start position
# "G" = goal (end point)

# 2. Grid class
class Grid:
    def __init__(self, rows: int = 8, cols: int = 16):
        self.rows = rows
        self.cols = cols
        # Initialize grid with all "."
        self.grid: List[List[str]] = [["." for _ in range(cols)] for _ in range(rows)]

    def place(self, row: int, col: int, tile: str) -> None:
        """Places a tile at that position"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row][col] = tile

    def get(self, row: int, col: int) -> str:
        """Returns tile at that position"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return ""

    def to_list(self) -> List[List[str]]:
        """Returns the full grid as List[List[str]]"""
        return [row[:] for row in self.grid]

    def from_list(self, data: List[List[str]]) -> None:
        """Loads a grid from List[List[str]]"""
        self.grid = [row[:] for row in data]
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    def find(self, tile: str) -> List[Tuple[int, int]]:
        """Returns list of (row,col) where that tile exists"""
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == tile:
                    positions.append((r, c))
        return positions

# 3. Helper functions
def make_basic_level() -> Grid:
    """
    Returns a Grid with:
    - bottom row (row 7) all "#" (floor)
    - "P" at row 6, col 0
    - "G" at row 6, col 15
    """
    grid = Grid()
    # Bottom row all walls
    for col in range(16):
        grid.place(7, col, "#")
    # Player start
    grid.place(6, 0, "P")
    # Goal
    grid.place(6, 15, "G")
    return grid

def make_gap_level() -> Grid:
    """
    Same as basic but with gaps at cols 5-6 and cols 10-11
    """
    grid = Grid()
    # Bottom row with gaps at 5-6 and 10-11
    for col in range(16):
        if col not in [5, 6, 10, 11]:
            grid.place(7, col, "#")
    # Player start
    grid.place(6, 0, "P")
    # Goal
    grid.place(6, 15, "G")
    return grid

# 4. Unit tests
if __name__ == "__main__":
    # Test 1: place P and G, check find() returns correct positions
    try:
        grid = Grid()
        grid.place(2, 3, "P")
        grid.place(5, 10, "G")

        p_positions = grid.find("P")
        g_positions = grid.find("G")

        assert (2, 3) in p_positions, "P should be at (2, 3)"
        assert (5, 10) in g_positions, "G should be at (5, 10)"
        assert len(p_positions) == 1, "Should have exactly one P"
        assert len(g_positions) == 1, "Should have exactly one G"

        print("✓ Test 1 PASS: place P and G, find() returns correct positions")
    except AssertionError as e:
        print(f"✗ Test 1 FAIL: {e}")

    # Test 2: make_basic_level() has P at (6,0) and G at (6,15)
    try:
        grid = make_basic_level()

        assert grid.get(6, 0) == "P", "P should be at (6, 0)"
        assert grid.get(6, 15) == "G", "G should be at (6, 15)"
        assert grid.get(7, 0) == "#", "Bottom row should be floor"
        assert grid.get(7, 15) == "#", "Bottom row should be floor"

        print("✓ Test 2 PASS: make_basic_level() has P at (6,0) and G at (6,15)")
    except AssertionError as e:
        print(f"✗ Test 2 FAIL: {e}")

    # Test 3: from_list(grid.to_list()) returns identical grid
    try:
        grid1 = make_gap_level()
        grid1.place(3, 7, "$")
        grid1.place(4, 8, "^")

        # Export and re-import
        data = grid1.to_list()
        grid2 = Grid()
        grid2.from_list(data)

        # Check all tiles match
        for r in range(8):
            for c in range(16):
                assert grid1.get(r, c) == grid2.get(r, c), f"Mismatch at ({r},{c})"

        print("✓ Test 3 PASS: from_list(grid.to_list()) returns identical grid")
    except AssertionError as e:
        print(f"✗ Test 3 FAIL: {e}")
