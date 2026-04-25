"""
FastAPI application for the LevelForge Environment.

Endpoints:
    - GET /health: Health check
    - POST /reset: Reset the environment with task_name and personality
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /curriculum_level: Get current curriculum level and avg reward
    - GET /scenarios: Get all curriculum levels
"""

import sys
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
from scenarios import CURRICULUM_LEVELS


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

# Mount static files - use path relative to project root for Docker compatibility
for static_path in ["/app/server/static", "server/static", os.path.join(_SERVER_DIR, "static")]:
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
        break

# Single environment instance
_env: Optional[LevelTrailEnvironment] = None


def get_env() -> LevelTrailEnvironment:
    global _env
    if _env is None:
        _env = LevelTrailEnvironment()
    return _env


@app.get("/")
async def root():
    """Serve the web UI."""
    possible_paths = [
        "/app/server/static/index.html",      # Docker absolute path
        "server/static/index.html",            # Relative from /app
        os.path.join(_SERVER_DIR, "static", "index.html"),  # Absolute server dir
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return FileResponse(path)
    return {"message": "LevelForge Environment API", "docs": "/docs"}

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


@app.get("/curriculum_level")
def curriculum_level():
    """Get current curriculum level based on rolling average reward."""
    env = get_env()
    return env.get_current_curriculum_info()


@app.get("/scenarios")
def scenarios():
    """Get all available curriculum levels."""
    return {"curriculum_levels": CURRICULUM_LEVELS}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
