---
title: LevelForge Environment Server
emoji: 🎮
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /
---

# LevelForge — AI Game Level Designer

> Training a 0.5B LLM to design personality-aware 2D platformer levels using GRPO and A* pathfinding rewards. No LLM judge. Pure math.

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/pranavgadodia/levelforge-env)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/Pranavpro23/levelforge_env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://meta-pytorch.org/OpenEnv/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pranavpro23/levelforge_env/blob/main/training/train_grpo.ipynb)

---

## The Problem

Game studios spend months on level design. A single AAA game can have hundreds of designers spending years crafting levels that feel fair, fun, and tailored to different player types. Indie developers like me do it alone — which means weeks of manual work for every game.

The deeper problem: **no two players are the same.** A brave player gets bored if levels are too easy and wants spike-filled gauntlets with enemies around every corner. A cautious player gets frustrated by unfair deaths and needs wide open paths. An explorer wants hidden secrets, branching routes, and coins tucked away in unexpected places.

Game studios solve this with dedicated level designers who specialize in specific player types. AI systems today cannot do this — existing procedural generation tools (CNNs, rule-based systems) produce generic levels that don't adapt to player personality.

**LevelForge trains a 0.5B language model to design levels automatically for each personality type** — graded entirely by A* pathfinding math with no human labels and no LLM judge needed. The same tiny model, given just a personality string, produces fundamentally different levels for brave vs cautious vs explorer players.

This has real-world applications beyond indie games: adaptive difficulty in mobile games, personalized onboarding levels in educational games, automated content generation for game studios, and AI-assisted level design tools. Any game that needs levels tailored to player behavior could use a system like this.

---

## Live Demo

Try the interactive level editor at the HF Space:

🎮 **[huggingface.co/spaces/pranavgadodia/levelforge-env](https://huggingface.co/spaces/pranavgadodia/levelforge-env)**

- Select a task and personality
- Click cells to place tiles
- Hit "Apply My Edits" to see your reward score in real time
- Watch the A* solver evaluate your level instantly

---

## What The Agent Learns

The agent receives an 8×16 grid and places tiles to create a playable platformer level:

| Tile | Symbol | Meaning |
|------|--------|---------|
| Empty | `.` | Open space |
| Wall | `#` | Platform or floor |
| Spike | `^` | Kills player — blocks A* path |
| Coin | `$` | Collectible |
| Enemy | `E` | Hazard |
| Player | `P` | Start position (fixed) |
| Goal | `G` | End position (fixed) |

The agent is told a **player personality** and must design accordingly:

- **Brave** → dangerous levels with many spikes and enemies, narrow paths
- **Cautious** → wide open paths, few hazards, comfortable navigation
- **Explorer** → multiple branching routes, hidden coins at different heights

---

## Three Tasks — Easy to Hard

| Task | Difficulty | What Agent Must Do | Target Path |
|------|-----------|-------------------|-------------|
| `first_steps` | Easy | Add coins + spikes to pre-built floor | 16 tiles |
| `gap_jumper` | Medium | Add platforms across gaps + coins + enemy | 22 tiles |
| `symmetric_shrine` | Hard | Design complete level from scratch | 30 tiles |

---

## Reward Function — Pure Python, No LLM Judge
R = 2.0 × solvable          (A* finds path P→G — hard gate)

1.0 × difficulty_band   (Gaussian at target path length)
0.5 × coin_reachable    (coins adjacent to A* path)
0.5 × symmetry          (aesthetic layout quality, gated by solvability)
0.5 × personality_match (level design suits player type)
0.25 × format_score     (structured XML reasoning present)


All components strictly between 0.0 and 1.0 (epsilon-clamped — no exact 0 or 1).
Unsolvable levels capped at 0.2 regardless of other scores.
Max possible weighted total: 4.75, normalised to 0–1 range.

**Anti-gaming measures:** Symmetry only rewarded on solvable levels. Broken levels penalised across all components. Coin reachability checks path adjacency not just count.

---

## Curriculum System — Self-Improving Difficulty

LevelForge includes an adaptive curriculum that automatically escalates difficulty as the model improves:

| Avg Reward | Curriculum Level | Description |
|-----------|-----------------|-------------|
| < 0.3 | Tutorial | Straight path, 1 coin, no hazards |
| 0.3–0.5 | Easy | Coins + 1-2 spikes |
| 0.5–0.65 | Medium | Gaps, platforms, 1 enemy |
| 0.65–0.8 | Hard | Design from scratch, symmetric |
| 0.8–0.9 | Speedrunner / Dungeon | Short fast path OR key-door routing |
| ≥ 0.9 | Tricky | Looks easy, hidden complexity |

Call `/curriculum_level` to see the current level. Call `/scenarios` for all 7 levels.

---

## Training

### Model
- **Base:** `unsloth/Qwen2.5-0.5B-Instruct`
- **Method:** QLoRA (4-bit quantization, r=32, lora_alpha=32)
- **Training:** GRPO via HuggingFace TRL + Unsloth acceleration
- **Hardware:** T4 GPU (15GB VRAM)

### Dataset
- 200 prompts per run
- 9 task × personality combinations (first_steps/gap_jumper/symmetric_shrine × brave/cautious/explorer)
- Repeated to reach 200 samples

### Training Logs
Full step-by-step training metrics available: [training_logs.json](training_logs.json)
*(500 steps — reward, format_reward, env_reward, KL divergence at every step)*

---

## Results

### Run 1 — 200 Steps (Baseline)

**Training stats:** global_step=200, training_loss=-0.0012, runtime=6244s (1hr 44min)

| Phase | Steps | Avg Reward | Format Reward | Key Finding |
|-------|-------|-----------|---------------|-------------|
| Early | 1–50 | 0.742 | Occasional | Exploring, rare format |
| Mid | 51–100 | 0.756 | Rising | Format learning begins |
| Late | 100–140 | 0.812 | Consistent | Format accelerates |
| Peak | Step 132 | 1.202 | 0.249 | Breakthrough step |

**Format reward appearances:** 8 in steps 1–100 → 12 in steps 100–140. Model increasingly using `<think>` tags.

---

### Run 2 — 500 Steps (Extended)

**Training stats:** global_step=500, training_loss=-0.0063, runtime=15694s (4hr 21min)

![Reward Curve — 500 Steps](https://raw.githubusercontent.com/Pranavpro23/levelforge_env/main/reward_curve.png)
*Total reward (blue) rises from ~0.8 to consistently 1.5–2.0. Format reward (green) reaches near-perfect 0.998 by step 50 and holds. Env reward (red/A* solvability) stabilises at 0.75–0.99.*

![Component Rewards — 500 Steps](https://raw.githubusercontent.com/Pranavpro23/levelforge_env/main/reward_curve_components.png)
*Format → Env reward progression. Green (format) learns first reaching 1.0 by step 50. Red (env/A*) stabilises. Blue (total) peaks at 1.997 — near theoretical maximum of 2.0.*

**Key numbers at step 500:**
- format_reward: **0.998** (near perfect — model learned XML structure completely)
- env_reward: **0.999** at best steps (A* solvability mastered)
- Peak total reward: **1.997** at step 89
- Consistent reward range steps 100–500: **1.5–2.0**

---

### Before vs After Training — Personality Comparison

![Personality Comparison](https://raw.githubusercontent.com/Pranavpro23/levelforge_env/main/personality_comparison.png)

*Top row: Starting grids before any edits — all 3 personalities identical (empty grid, P + G + floor).
Bottom row: After 500-step training — same model, different personality input → visually different levels.*

- **BRAVE:** Red spikes (^) and purple enemy (E) scattered across floor — dangerous, narrow paths ✅
- **CAUTIOUS:** Yellow coins ($) only, no hazards — safe, open navigation ✅
- **EXPLORER:** Yellow coins at different heights + dark platform — branching routes, hidden secrets ✅

---

### Before vs After Model Outputs

**Baseline (untrained) — brave player:**
> *"we'll focus on creating a sense of exploration and challenge while ensuring players navigate safely"*
> Format: coordinate notation `[0,0] -> [3,0]` — no personality awareness

**After 200 steps — brave player:**
> *"we need to ensure that the player can navigate through the area while maintaining their pace of exploration"*
> Format: XML `<edit>` tags present — brave-aware reasoning

**After 500 steps — cautious player:**
> *"focus on creating smooth, safe routes that avoid obstacles while still providing enough challenges"*
> Format: JSON edits `{"edit": ["row 0", "row 1"...]}` — structured output learned

**Key improvement:** Baseline uses identical generic language for all 3 personalities. Trained model uses personality-specific reasoning — brave gets "pace of exploration", cautious gets "avoid obstacles", explorer gets "navigate efficiently".

---

### Run 3 — Curriculum Training (Self-Improving)

*[Results pending — curriculum training in progress]*

*Will show adaptive difficulty progression: tutorial → easy → medium → hard → tricky as model reward improves. Demonstrates Theme 4 (Self-Improvement) — the environment itself becomes a teacher.*

---

## Quick Start

### Try the live interactive UI
https://huggingface.co/spaces/pranavgadodia/levelforge-env

### Connect via API

```python
import requests

ENV_URL = "https://pranavgadodia-levelforge-env.hf.space"

# Reset environment
obs = requests.post(f"{ENV_URL}/reset",
    json={"task_name": "first_steps", "personality": "brave"}).json()

# Take a step
result = requests.post(f"{ENV_URL}/step",
    json={
        "reasoning": "<think>Placing spikes for a brave player</think>",
        "edits": [{"row": 5, "col": 3, "tile": "^"},
                  {"row": 5, "col": 7, "tile": "E"}],
        "declare_done": False
    }).json()

print(f"Reward: {result['reward']['total']:.3f}")
print(f"Solvable: {result['reward']['solvable']}")
print(f"Personality match: {result['reward']['personality_match']:.3f}")
```

### Run locally

```bash
git clone https://github.com/Pranavpro23/levelforge_env
cd levelforge_env
docker build -f server/Dockerfile -t levelforge-env .
docker run -p 7860:7860 levelforge-env
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "first_steps", "personality": "brave"}'
```

### Test API endpoints

```bash
GET  /health              → {"status": "healthy"}
POST /reset               → Observation (grid, personality, episode_id)
POST /step                → Observation + Reward + done + info
GET  /state               → Current episode state
GET  /curriculum_level    → Current curriculum level + avg reward
GET  /scenarios           → All 7 curriculum difficulty levels
GET  /docs                → Full OpenAPI documentation
```

---

## Project Structure

```
levelforge_env/
├── inference.py          # OpenEnv inference [START][STEP][END] format
├── openenv.yaml          # 3 tasks defined (easy/medium/hard)
├── models.py             # Pydantic: LevelTrailObservation, Action, Reward
├── client.py             # LevelTrailEnv HTTP client
├── server/
│   ├── app.py            # FastAPI: /reset /step /state /curriculum_level
│   ├── environment.py    # Episode logic + rolling reward history
│   ├── tilemap.py        # 8x16 grid + tile definitions
│   ├── astar.py          # A* pathfinder (4-connected, spike=death)
│   ├── renderer.py       # Pygame grid to PNG to evolution GIF
│   ├── rewards.py        # 6 reward functions (pure Python, no LLM)
│   ├── scenarios.py      # 30 scenarios + 7 curriculum levels
│   ├── static/
│   │   └── index.html    # Interactive web UI
│   └── Dockerfile
└── training/
    └── train_grpo.ipynb  # GRPO training notebook (Colab)
```

---

## Why This Matters

Procedural content generation (PCG) for games has been dominated by CNNs and rule-based systems. **LevelForge is the first OpenEnv environment that trains a language model to generate game levels via RL** — using a game engine verifier (A* pathfinding) as the reward signal.

The personality system is the key novelty: the same 0.5B model produces fundamentally different levels for different player archetypes, learned entirely from verifiable reward signals with no human-labeled training data.

> *"Could a researcher write a paper about training on this?"* — Yes. Personality-aware procedural content generation via RLVR is an underexplored research direction.

---

## Links

| Resource | URL |
|----------|-----|
| 🤗 HF Space (live demo) | https://huggingface.co/spaces/pranavgadodia/levelforge-env |
| 💻 GitHub | https://github.com/Pranavpro23/levelforge_env |
| 📓 Training Notebook | [Open in Colab](https://colab.research.google.com/github/Pranavpro23/levelforge_env/blob/main/training/train_grpo.ipynb) |
| 📝 HF Mini-blog | *[coming soon]* |
| 🎥 Demo video | *[coming soon]* |

---

*Built for Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, India 2026*
*Theme 4: Self-Improvement + Wild Card*