"""
Reward functions for the LevelForge environment.

Every function returns a float strictly between 0.0 and 1.0 (never exactly 0 or 1).
"""

import math
import sys
import os
from typing import List, Optional, Tuple
from collections import deque

# ---------------------------------------------------------------------------
# Epsilon-clamp helper
# ---------------------------------------------------------------------------

def clamp(value: float, epsilon: float = 1e-6) -> float:
    """Return value clamped to (epsilon, 1.0 - epsilon)."""
    return max(epsilon, min(1.0 - epsilon, float(value)))


# ---------------------------------------------------------------------------
# Internal imports (astar lives in the same server/ package)
# ---------------------------------------------------------------------------

# When run as a script __package__ is None, so we add server/ to sys.path.
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from astar import find_path, path_length as _path_length, is_solvable  # noqa: E402
from tilemap import Grid, make_basic_level  # noqa: E402


# ---------------------------------------------------------------------------
# models import — models.py lives one level up from server/
# ---------------------------------------------------------------------------

_ROOT_DIR = os.path.dirname(_SERVER_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from models import LevelTrailReward  # noqa: E402


# ---------------------------------------------------------------------------
# FUNCTION 1 — solvable_reward
# ---------------------------------------------------------------------------

def solvable_reward(grid: List[List[str]]) -> float:
    """
    0.999 if A* finds a path from P to G, 0.001 otherwise.
    Always returns a value strictly between 0 and 1.
    """
    path = find_path(grid)
    raw = 0.999 if path is not None else 0.001
    return clamp(raw)


# ---------------------------------------------------------------------------
# FUNCTION 2 — difficulty_band_reward
# ---------------------------------------------------------------------------

def difficulty_band_reward(grid: List[List[str]], target_path_len: int) -> float:
    """
    Gaussian reward centred on target_path_len with sigma=5.
    Always returns a value strictly between 0 and 1.
    """
    actual = _path_length(grid)
    raw = math.exp(-((actual - target_path_len) / 5) ** 2)
    return clamp(raw)


# ---------------------------------------------------------------------------
# FUNCTION 3 — coin_reachable_reward
# ---------------------------------------------------------------------------

def coin_reachable_reward(grid: List[List[str]]) -> float:
    """
    Fraction of coins that are 4-connected adjacent to any tile on the A* path.
    Always returns a value strictly between 0 and 1.
    """
    path = find_path(grid)
    if path is None:
        return clamp(0.001)

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Build a set of all path positions for fast lookup
    path_set = set(path)

    # Count total coins in the grid
    total_coins = sum(
        1
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] == "$"
    )

    if total_coins == 0:
        # No coins to collect — treat as fully satisfied (clamped near 1)
        return clamp(0.999)

    # Count coins adjacent (4-connected) to any path tile
    adjacent_coins = 0
    counted = set()
    for (pr, pc) in path:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if (nr, nc) not in counted and 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == "$":
                    adjacent_coins += 1
                    counted.add((nr, nc))

    raw = adjacent_coins / max(total_coins, 1)
    return clamp(raw)


# ---------------------------------------------------------------------------
# FUNCTION 4 — symmetry_reward
# ---------------------------------------------------------------------------

def symmetry_reward(grid: List[List[str]]) -> float:
    """
    Horizontal (left-right) symmetry of non-empty, non-P, non-G tiles.
    Gate: unsolvable levels get no symmetry credit.
    Always returns a value strictly between 0 and 1.
    """
    # Gate: unsolvable levels get no symmetry credit
    if not is_solvable(grid):
        return clamp(0.001)

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    flipped = [row[::-1] for row in grid]

    matching = 0
    total = 0

    for r in range(rows):
        for c in range(cols):
            tile = grid[r][c]
            # Only count meaningful tiles (not P, G, or empty)
            if tile not in ("P", "G", "."):
                total += 1
                if grid[r][c] == flipped[r][c]:
                    matching += 1

    if total == 0:
        return clamp(0.001)  # empty grid, no meaningful symmetry

    raw = matching / total
    return clamp(raw)


# ---------------------------------------------------------------------------
# FUNCTION 5 — personality_match_reward
# ---------------------------------------------------------------------------

def personality_match_reward(grid: List[List[str]], personality: str) -> float:
    """
    Reward based on how well the level design matches the target personality:
      brave    — more hazards is better
      cautious — more open space (empty tiles) is better
      explorer — more BFS branch-points from P is better
      unknown  — neutral 0.5
    Always returns a value strictly between 0 and 1.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Count hazards (E and ^) and empty tiles
    hazards = sum(
        1
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] in ("E", "^")
    )
    empty_count = sum(
        1
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] == "."
    )

    p = personality.lower()

    if p == "brave":
        raw = min(hazards / 8, 0.999)

    elif p == "cautious":
        raw = min(empty_count / 100, 0.999)

    elif p == "explorer":
        # BFS from P, count branch points (cells with >1 passable neighbour
        # reachable from P that haven't been visited via a single path)
        passable = {".", "$", "E", "P", "G"}  # tiles the player can enter

        # Find P
        start: Optional[Tuple[int, int]] = None
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "P":
                    start = (r, c)
                    break
            if start:
                break

        if start is None:
            return clamp(0.001)

        visited: set = set()
        queue: deque = deque([start])
        visited.add(start)
        branch_points = 0

        while queue:
            r, c = queue.popleft()
            passable_neighbours = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in passable:
                    passable_neighbours.append((nr, nc))

            # A branch point has more than one passable neighbour
            if len(passable_neighbours) > 1:
                branch_points += 1

            for nb in passable_neighbours:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        raw = min(branch_points / 4, 0.999)

    else:
        # unknown personality
        raw = 0.5

    return clamp(raw)


# ---------------------------------------------------------------------------
# FUNCTION 6 — format_reward
# ---------------------------------------------------------------------------

def format_reward(reasoning: str) -> float:
    """
    Awards 0.499 per found thinking tag (<think> and </think>).
    Max raw = 0.998 when both tags present.
    Always returns a value strictly between 0 and 1.
    """
    found_tags = (1 if "<think>" in reasoning else 0) + (1 if "</think>" in reasoning else 0)
    raw = found_tags * 0.499
    return clamp(raw)


# ---------------------------------------------------------------------------
# MAIN FUNCTION — compute_total
# ---------------------------------------------------------------------------

def compute_total(
    grid: List[List[str]],
    reasoning: str,
    personality: str,
    target_path_len: int,
) -> LevelTrailReward:
    """
    Compute all six reward components, apply a broken-level penalty, normalise,
    and return a LevelTrailReward with all fields strictly between 0 and 1.
    """
    solvable         = solvable_reward(grid)
    difficulty_band  = difficulty_band_reward(grid, target_path_len)
    coin_reachable   = coin_reachable_reward(grid)
    symmetry         = symmetry_reward(grid)
    personality_match = personality_match_reward(grid, personality)
    format_score     = format_reward(reasoning)

    weighted_sum = (
        2.00 * solvable
        + 1.00 * difficulty_band
        + 0.50 * coin_reachable
        + 0.50 * symmetry
        + 0.50 * personality_match
        + 0.25 * format_score
    )

    # Broken-level penalty: cap weighted_sum if level is not solvable
    if solvable < 0.5:
        weighted_sum = min(weighted_sum, 0.95)

    normalised = weighted_sum / 4.75
    total = clamp(normalised)

    return LevelTrailReward(
        solvable=solvable,
        difficulty_band=difficulty_band,
        coin_reachable=coin_reachable,
        symmetry=symmetry,
        personality_match=personality_match,
        format_score=format_score,
        total=total,
    )


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tilemap import Grid, make_basic_level

    all_passed = True

    def _result(label: str, passed: bool, value) -> None:
        global all_passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: {value}")
        if not passed:
            all_passed = False

    print("=" * 60)
    print("rewards.py — running 8 tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: solvable_reward on a solvable grid
    # ------------------------------------------------------------------
    try:
        g = make_basic_level().to_list()
        result = solvable_reward(g)
        passed = 0.0 < result < 1.0
        _result("Test 1  solvable_reward (solvable grid)", passed, result)
    except Exception as exc:
        _result("Test 1  solvable_reward (solvable grid)", False, exc)

    # ------------------------------------------------------------------
    # Test 2: solvable_reward on an unsolvable grid
    # ------------------------------------------------------------------
    try:
        bad = Grid()
        bad.place(6, 0, "P")
        bad.place(6, 15, "G")
        # Full vertical wall — no path around it
        for row in range(8):
            bad.place(row, 8, "#")
        result = solvable_reward(bad.to_list())
        passed = 0.0 < result < 1.0
        _result("Test 2  solvable_reward (unsolvable grid)", passed, result)
    except Exception as exc:
        _result("Test 2  solvable_reward (unsolvable grid)", False, exc)

    # ------------------------------------------------------------------
    # Test 3: difficulty_band_reward when actual == target (never exactly 1)
    # ------------------------------------------------------------------
    try:
        g = make_basic_level().to_list()
        # basic level path length is 16
        result = difficulty_band_reward(g, 16)
        passed = result < 1.0
        _result("Test 3  difficulty_band_reward (exact target)", passed, result)
    except Exception as exc:
        _result("Test 3  difficulty_band_reward (exact target)", False, exc)

    # ------------------------------------------------------------------
    # Test 4: personality_match brave with 5 hazards → result > 0.5
    # ------------------------------------------------------------------
    try:
        g = make_basic_level()
        # Add 5 spikes on row 5
        for c in range(5):
            g.place(5, c + 1, "^")
        result = personality_match_reward(g.to_list(), "brave")
        passed = result > 0.5
        _result("Test 4  personality_match brave (5 hazards)", passed, result)
    except Exception as exc:
        _result("Test 4  personality_match brave (5 hazards)", False, exc)

    # ------------------------------------------------------------------
    # Test 5: personality_match cautious with 5 hazards → result < 0.3
    # A dense grid (mostly walls) ensures very few empty tiles,
    # so empty_count / 100 is well below 0.3.
    # ------------------------------------------------------------------
    try:
        dense = Grid()
        # Fill the entire grid with walls
        for r in range(8):
            for c in range(16):
                dense.place(r, c, "#")
        # Carve out exactly 10 empty cells (empty_count / 100 = 0.1 < 0.3)
        for c in range(10):
            dense.place(3, c, ".")
        # Place P and G (non-empty, excluded from empty count)
        dense.place(6, 0, "P")
        dense.place(6, 15, "G")
        # Add 5 spikes
        for c in range(5):
            dense.place(5, c + 1, "^")
        result = personality_match_reward(dense.to_list(), "cautious")
        passed = result < 0.3
        _result("Test 5  personality_match cautious (5 hazards, dense grid)", passed, result)
    except Exception as exc:
        _result("Test 5  personality_match cautious (5 hazards, dense grid)", False, exc)

    # ------------------------------------------------------------------
    # Test 6: format_reward with both <think> and </think> tags
    # ------------------------------------------------------------------
    try:
        reasoning = "<think>This is my reasoning</think>"
        result = format_reward(reasoning)
        passed = result < 1.0
        _result("Test 6  format_reward (both tags present)", passed, result)
    except Exception as exc:
        _result("Test 6  format_reward (both tags present)", False, exc)

    # ------------------------------------------------------------------
    # Test 7: compute_total on a good (solvable) level → 0 < total < 1
    # ------------------------------------------------------------------
    try:
        g = make_basic_level()
        # Add a coin next to the path
        g.place(6, 5, "$")
        result = compute_total(
            g.to_list(),
            reasoning="<think>Place coin along path</think>",
            personality="brave",
            target_path_len=16,
        )
        passed = 0.0 < result.total < 1.0
        _result("Test 7  compute_total (good level)", passed, result.total)
    except Exception as exc:
        _result("Test 7  compute_total (good level)", False, exc)

    # ------------------------------------------------------------------
    # Test 8: compute_total on a broken level (no path) → 0 < total < 0.25
    # ------------------------------------------------------------------
    try:
        bad = Grid()
        bad.place(6, 0, "P")
        bad.place(6, 15, "G")
        for row in range(8):
            bad.place(row, 8, "#")
        result = compute_total(
            bad.to_list(),
            reasoning="no tags here",
            personality="brave",
            target_path_len=16,
        )
        passed = 0.0 < result.total < 0.25
        _result("Test 8  compute_total (broken level)", passed, result.total)
    except Exception as exc:
        _result("Test 8  compute_total (broken level)", False, exc)

    print("=" * 60)
    print("All tests PASSED" if all_passed else "Some tests FAILED")
    print("=" * 60)
