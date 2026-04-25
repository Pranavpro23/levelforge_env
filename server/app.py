"""
FastAPI application for the LevelForge Environment.

Endpoints:
    - GET /health: Health check
    - POST /reset: Reset the environment with task_name and personality
    - POST /step: Execute an action
    - GET /state: Get current environment state
"""

import sys
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Path setup
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SERVER_DIR)

if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from models import LevelTrailObservation, LevelTrailAction, LevelTrailReward
from environment import LevelTrailEnvironment


# Request/Response models
class ResetRequest(BaseModel):
    task_name: str
    personality: str


class StepResponse(BaseModel):
    observation: LevelTrailObservation
    reward: LevelTrailReward
    done: bool
    info: dict


# Create FastAPI app
app = FastAPI(
    title="LevelForge Environment",
    description="OpenEnv-compatible environment for training LLMs to design platformer levels",
    version="1.0.0"
)

# Single environment instance
_env: Optional[LevelTrailEnvironment] = None


def get_env() -> LevelTrailEnvironment:
    global _env
    if _env is None:
        _env = LevelTrailEnvironment()
    return _env


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset", response_model=LevelTrailObservation)
def reset(request: ResetRequest):
    """Reset the environment with a new episode."""
    env = get_env()
    try:
        obs = env.reset(request.task_name, request.personality)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(action: LevelTrailAction):
    """Execute one step in the environment."""
    env = get_env()
    if env._episode_id is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")

    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Get current environment state."""
    env = get_env()
    return env.state()




def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
