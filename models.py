"""
Data models for the LevelForge Environment.

The LevelForge environment trains an LLM to design 2D platformer levels.
"""

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Fallback for testing without openenv installed
    from pydantic import BaseModel
    Action = BaseModel
    Observation = BaseModel

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# CLASS 1 — LevelTrailObservation
class LevelTrailObservation(Observation):
    """Observation returned to the agent at each step"""
    episode_id: str
    step: int  # current step number, starts at 0
    grid: List[List[str]]  # the current level grid, 8 rows x 16 cols
    target_path_len: int  # what difficulty the AI should aim for
    target_coin_count: int  # how many coins should be reachable
    personality: str  # one of: "brave", "cautious", "explorer"
    task_name: str  # one of: "first_steps", "gap_jumper", "symmetric_shrine"
    last_player_result: Optional[Dict[str, Any]] = None  # None on first step, then has keys: reached_goal, path_len, coins_collected, deaths
    feedback_tags: List[str] = Field(default_factory=list)  # hints like "unreachable_goal", "too_easy", "cheap_spike_kill"
    max_steps: int = 8  # always 8

# CLASS 2 — LevelTrailAction
class LevelTrailAction(Action):
    """Action submitted by the agent"""
    reasoning: str = Field(..., max_length=400)  # AI explains its thinking, max 400 chars
    edits: List[Dict[str, Any]] = Field(default_factory=list)  # each dict has: row: int, col: int, tile: str
    declare_done: bool = False  # AI can say it's finished early

# CLASS 3 — LevelTrailReward
class LevelTrailReward(BaseModel):
    """Reward breakdown for transparency"""
    solvable: float  # 0.0 or 1.0 — did A* find a path?
    difficulty_band: float  # 0.0-1.0 — how close to target path length?
    coin_reachable: float  # 0.0-1.0 — fraction of coins on the path
    symmetry: float  # 0.0-1.0 — how symmetric is the level?
    personality_match: float  # 0.0-1.0 — does level suit the personality?
    format_score: float  # 0.0-1.0 — did AI use proper reasoning format?
    total: float  # weighted sum of all above

# LevelTrailState
class LevelTrailState(BaseModel):
    """Internal environment state"""
    episode_id: str
    step: int
    task_name: str
    personality: str
    done: bool
    total_reward_so_far: float

# Quick validation
if __name__ == "__main__":
    # Test instantiation
    obs = LevelTrailObservation(
        episode_id="test-123",
        step=0,
        grid=[["." for _ in range(16)] for _ in range(8)],
        target_path_len=16,
        target_coin_count=3,
        personality="brave",
        task_name="first_steps"
    )

    action = LevelTrailAction(
        reasoning="Adding coins along the path for brave player",
        edits=[{"row": 3, "col": 5, "tile": "$"}]
    )

    reward = LevelTrailReward(
        solvable=1.0,
        difficulty_band=0.8,
        coin_reachable=0.6,
        symmetry=0.5,
        personality_match=0.7,
        format_score=1.0,
        total=4.6
    )

    state = LevelTrailState(
        episode_id="test-123",
        step=0,
        task_name="first_steps",
        personality="brave",
        done=False,
        total_reward_so_far=0.0
    )

    print("models.py OK")
