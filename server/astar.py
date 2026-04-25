from typing import List, Tuple, Optional
import heapq

def find_path(grid: List[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """
    A* search from P to G.
    Movement: 4-connected (up/down/left/right only)
    Blocked tiles: "#" (wall) and "^" (spike)
    Returns: list of (row,col) tuples showing the path, or None if no path exists
    The returned path includes the start P and end G positions
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find P and G
    start = None
    goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "P":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)

    if start is None or goal is None:
        return None

    # A* algorithm
    def heuristic(pos: Tuple[int, int]) -> int:
        """Manhattan distance"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # Priority queue: (f_score, counter, position)
    counter = 0
    open_set = [(heuristic(start), counter, start)]
    counter += 1

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Check all 4 neighbors (up, down, left, right)
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            # Check bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = (nr, nc)
            tile = grid[nr][nc]

            # Check if blocked by wall or spike
            if tile in ["#", "^"]:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, counter, neighbor))
                counter += 1

    return None

def path_length(grid: List[List[str]]) -> int:
    """Returns length of path from find_path(), or 999 if no path"""
    path = find_path(grid)
    if path is None:
        return 999
    return len(path)

def is_solvable(grid: List[List[str]]) -> bool:
    """Returns True if find_path() returns a path (not None)"""
    return find_path(grid) is not None

# 4. Unit tests
if __name__ == "__main__":
    from tilemap import Grid, make_basic_level

    # Test 1: basic level with clear path → is_solvable returns True
    try:
        grid = make_basic_level()
        grid_data = grid.to_list()

        assert is_solvable(grid_data) == True, "Basic level should be solvable"
        print("✓ Test 1 PASS: basic level with clear path → is_solvable returns True")
    except AssertionError as e:
        print(f"✗ Test 1 FAIL: {e}")

    # Test 2: level with wall blocking path → is_solvable returns False
    try:
        grid = Grid()
        grid.place(6, 0, "P")
        grid.place(6, 15, "G")
        # Add vertical wall blocking the path
        for row in range(8):
            grid.place(row, 8, "#")

        grid_data = grid.to_list()
        assert is_solvable(grid_data) == False, "Blocked level should not be solvable"
        print("✓ Test 2 PASS: level with wall blocking path → is_solvable returns False")
    except AssertionError as e:
        print(f"✗ Test 2 FAIL: {e}")

    # Test 3: level with spike blocking only path → is_solvable returns False
    try:
        grid = Grid()
        grid.place(6, 0, "P")
        grid.place(6, 15, "G")
        # Add vertical spike wall blocking the path
        for row in range(8):
            grid.place(row, 8, "^")

        grid_data = grid.to_list()
        assert is_solvable(grid_data) == False, "Spike-blocked level should not be solvable"
        print("✓ Test 3 PASS: level with spike blocking only path → is_solvable returns False")
    except AssertionError as e:
        print(f"✗ Test 3 FAIL: {e}")

    # Test 4: path_length on basic level returns correct number
    try:
        grid = make_basic_level()
        grid_data = grid.to_list()

        length = path_length(grid_data)
        # From (6,0) to (6,15) in straight line = 16 positions
        assert length == 16, f"Path length should be 16, got {length}"
        print(f"✓ Test 4 PASS: path_length on basic level returns correct number ({length})")
    except AssertionError as e:
        print(f"✗ Test 4 FAIL: {e}")
