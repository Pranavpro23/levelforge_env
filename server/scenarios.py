from typing import List, Dict, Any
import random
from tilemap import Grid, make_basic_level, make_gap_level


CURRICULUM_LEVELS = [
    {
        "level": 0,
        "name": "tutorial",
        "description": "Straight path, just add 1 coin",
        "target_path_len": 12,
        "target_coin_count": 1,
        "max_spikes": 0,
        "max_enemies": 0,
        "starting_grid": "basic"
    },
    {
        "level": 1,
        "name": "easy",
        "description": "Add coins and 1-2 spikes",
        "target_path_len": 16,
        "target_coin_count": 3,
        "max_spikes": 2,
        "max_enemies": 0,
        "starting_grid": "basic"
    },
    {
        "level": 2,
        "name": "medium",
        "description": "Gaps in floor, add platforms",
        "target_path_len": 22,
        "target_coin_count": 4,
        "max_spikes": 3,
        "max_enemies": 1,
        "starting_grid": "gap"
    },
    {
        "level": 3,
        "name": "hard",
        "description": "Design from scratch, symmetric",
        "target_path_len": 28,
        "target_coin_count": 5,
        "max_spikes": 4,
        "max_enemies": 2,
        "starting_grid": "empty"
    },
    {
        "level": 4,
        "name": "speedrunner",
        "description": "Straight short path, minimal obstacles",
        "target_path_len": 10,
        "target_coin_count": 2,
        "max_spikes": 1,
        "max_enemies": 0,
        "starting_grid": "basic"
    },
    {
        "level": 5,
        "name": "dungeon",
        "description": "Add key K and door D tiles, agent must route player through key first",
        "target_path_len": 25,
        "target_coin_count": 3,
        "max_spikes": 3,
        "max_enemies": 2,
        "starting_grid": "empty"
    },
    {
        "level": 6,
        "name": "tricky",
        "description": "Looks easy but has hidden complexity, red herrings",
        "target_path_len": 20,
        "target_coin_count": 6,
        "max_spikes": 5,
        "max_enemies": 3,
        "starting_grid": "gap"
    }
]


def get_curriculum_level(reward_avg: float) -> dict:
    """
    Select curriculum level based on rolling average reward.
    Progressive difficulty scaling based on agent performance.
    """
    if reward_avg < 0.3:
        level_idx = 0  # tutorial
    elif reward_avg < 0.5:
        level_idx = 1  # easy
    elif reward_avg < 0.65:
        level_idx = 2  # medium
    elif reward_avg < 0.8:
        level_idx = 3  # hard
    elif reward_avg < 0.9:
        level_idx = random.choice([4, 5])  # speedrunner or dungeon
    else:
        level_idx = 6  # tricky

    return CURRICULUM_LEVELS[level_idx].copy()


def get_scenarios() -> List[Dict[str, Any]]:
    """
    Returns 30 training scenarios:
    - 15 first_steps (5 brave, 5 cautious, 5 explorer)
    - 10 gap_jumper (4 brave, 3 cautious, 3 explorer)
    - 5 symmetric_shrine (mix of personalities)
    """
    scenarios = []

    # --- FIRST_STEPS (15 scenarios) ---
    # 5 brave
    for _ in range(5):
        grid = make_basic_level()
        scenarios.append({
            "task_name": "first_steps",
            "personality": "brave",
            "starting_grid": grid.to_list(),
            "target_path_len": 16,
            "target_coin_count": 3
        })

    # 5 cautious
    for _ in range(5):
        grid = make_basic_level()
        scenarios.append({
            "task_name": "first_steps",
            "personality": "cautious",
            "starting_grid": grid.to_list(),
            "target_path_len": 16,
            "target_coin_count": 3
        })

    # 5 explorer
    for _ in range(5):
        grid = make_basic_level()
        scenarios.append({
            "task_name": "first_steps",
            "personality": "explorer",
            "starting_grid": grid.to_list(),
            "target_path_len": 16,
            "target_coin_count": 3
        })

    # --- GAP_JUMPER (10 scenarios) ---
    # 4 brave
    for _ in range(4):
        grid = make_gap_level()
        scenarios.append({
            "task_name": "gap_jumper",
            "personality": "brave",
            "starting_grid": grid.to_list(),
            "target_path_len": 22,
            "target_coin_count": 4
        })

    # 3 cautious
    for _ in range(3):
        grid = make_gap_level()
        scenarios.append({
            "task_name": "gap_jumper",
            "personality": "cautious",
            "starting_grid": grid.to_list(),
            "target_path_len": 22,
            "target_coin_count": 4
        })

    # 3 explorer
    for _ in range(3):
        grid = make_gap_level()
        scenarios.append({
            "task_name": "gap_jumper",
            "personality": "explorer",
            "starting_grid": grid.to_list(),
            "target_path_len": 22,
            "target_coin_count": 4
        })

    # --- SYMMETRIC_SHRINE (5 scenarios) ---
    # Empty grid with just P at (6,0) and G at (6,15)
    personalities = ["brave", "cautious", "explorer", "brave", "explorer"]

    for personality in personalities:
        grid = Grid(rows=8, cols=16)
        grid.place(6, 0, "P")
        grid.place(6, 15, "G")

        scenarios.append({
            "task_name": "symmetric_shrine",
            "personality": personality,
            "starting_grid": grid.to_list(),
            "target_path_len": 30,
            "target_coin_count": 6
        })

    return scenarios


if __name__ == "__main__":
    scenarios = get_scenarios()
    print(f"Created {len(scenarios)} scenarios")

    # Print breakdown
    task_counts = {}
    personality_counts = {}

    for scenario in scenarios:
        task = scenario["task_name"]
        personality = scenario["personality"]

        task_counts[task] = task_counts.get(task, 0) + 1
        personality_counts[personality] = personality_counts.get(personality, 0) + 1

    print("\nTask breakdown:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")

    print("\nPersonality breakdown:")
    for personality, count in personality_counts.items():
        print(f"  {personality}: {count}")
