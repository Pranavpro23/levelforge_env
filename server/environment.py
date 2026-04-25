"""
LevelTrailEnvironment — manages one episode at a time.
"""

import sys
import os
from uuid import uuid4
from random import choice
from typing import List, Dict, Any, Tuple, Optional

# ---------------------------------------------------------------------------
# Path setup — models.py lives one level up from server/
# ---------------------------------------------------------------------------

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SERVER_DIR)

if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from models import LevelTrailObservation, LevelTrailAction, LevelTrailReward  # noqa: E402
from scenarios import get_scenarios  # noqa: E402
from rewards import compute_total  # noqa: E402


class LevelTrailEnvironment:
    """Manages one LevelForge episode at a time."""

    max_concurrent_envs: int = 32

    def __init__(self) -> None:
        self._scenarios: List[Dict[str, Any]] = get_scenarios()
        self._episode_id: Optional[str] = None
        self._step: int = 0
        self._task_name: Optional[str] = None
        self._personality: Optional[str] = None
        self._grid: Optional[List[List[str]]] = None
        self._target_path_len: int = 0
        self._target_coin_count: int = 0
        self._done: bool = True
        self._total_reward: float = 0.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_name: str, personality: str) -> LevelTrailObservation:
        """Start a new episode for the given task_name / personality pair."""
        matching = [
            s for s in self._scenarios
            if s["task_name"] == task_name and s["personality"] == personality
        ]
        scenario = choice(matching)

        # Deep-copy the starting grid so the scenario template is never mutated
        self._grid = [row[:] for row in scenario["starting_grid"]]

        self._episode_id = str(uuid4())
        self._step = 0
        self._done = False
        self._total_reward = 0.0
        self._task_name = task_name
        self._personality = personality
        self._target_path_len = scenario["target_path_len"]
        self._target_coin_count = scenario["target_coin_count"]

        return LevelTrailObservation(
            episode_id=self._episode_id,
            step=0,
            grid=self._grid,
            target_path_len=self._target_path_len,
            target_coin_count=self._target_coin_count,
            personality=personality,
            task_name=task_name,
            last_player_result=None,
            feedback_tags=[],
            max_steps=8,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, action: LevelTrailAction
    ) -> Tuple[LevelTrailObservation, LevelTrailReward, bool, Dict[str, Any]]:
        """Apply one agent action and return (obs, reward, done, info)."""
        # Apply tile edits
        for edit in action.edits:
            row, col, tile = edit["row"], edit["col"], edit["tile"]
            self._grid[row][col] = tile

        self._step += 1

        # Compute reward
        reward = compute_total(
            self._grid,
            action.reasoning,
            self._personality,
            self._target_path_len,
        )
        self._total_reward += reward.total

        # Build feedback tags
        feedback_tags: List[str] = []
        if reward.solvable < 0.5:
            feedback_tags.append("unreachable_goal")
        if reward.difficulty_band < 0.3:
            feedback_tags.append("wrong_difficulty")
        if reward.personality_match < 0.3:
            feedback_tags.append("personality_mismatch")

        # Check terminal condition
        done = self._step >= 8 or action.declare_done
        self._done = done

        observation = LevelTrailObservation(
            episode_id=self._episode_id,
            step=self._step,
            grid=self._grid,
            target_path_len=self._target_path_len,
            target_coin_count=self._target_coin_count,
            personality=self._personality,
            task_name=self._task_name,
            last_player_result=None,
            feedback_tags=feedback_tags,
            max_steps=8,
        )

        return observation, reward, done, {"episode_id": self._episode_id}

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Return a snapshot of the current episode state."""
        return {
            "step": self._step,
            "task_name": self._task_name,
            "personality": self._personality,
            "done": self._done,
            "total_reward_so_far": self._total_reward,
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = LevelTrailEnvironment()
    obs = env.reset("first_steps", "brave")
    print(f"Reset: episode_id={obs.episode_id}, step={obs.step}, task={obs.task_name}")

    action = LevelTrailAction(reasoning="Testing edit", edits=[{"row": 5, "col": 5, "tile": "$"}])
    obs, reward, done, info = env.step(action)
    print(f"Step 1: step={obs.step}, reward.total={reward.total:.3f}, done={done}")

    state = env.state()
    print(f"State: {state}")
    print("environment.py OK")
