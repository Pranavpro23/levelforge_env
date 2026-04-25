from typing import List, Dict, Any
from tilemap import Grid, make_basic_level, make_gap_level


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
