# LevelForge — OpenEnv Hackathon Project

## What we're building
A game level designer environment where a small LLM (Qwen2.5-0.5B) learns via GRPO 
to design 2D platformer levels for a tiger-cub game called "The Last Waterfall".

## Theme
OpenEnv Hackathon Theme 4 (Self-Improvement) + Wild Card

Theme #4 - Self-Improvement
The focus here is to create environments where agents can learn to generate new
challenges, escalate difficulty, and improve through self-play or adaptive curricula.
Rather than optimizing fixed tasks, the goal is for agents to learn to drive their own
capability growth. The objective is recursive skill amplification.
Expected Outcome: an environment for improving self-play of a LLM over a defined
set of tasks
Example environments: Self-play negotiation arenas, auto-generated math/proof
tasks, evolving coding competitions, adaptive RL curricula.
Theme #5: Wild Card - Impress Us!
We do not want to limit your focus if your idea doesn’t fit the boxes above, we want
and WILL reward out of box tasks, please be creative but remember to add
submissions that meaningfully add value to LLM training on a certain task.
Guidelines for Problem Statement
●
●
●
It is NOT mandatory to choose the same problem statement as Round 1. Only
choose the same problem statement if it aligns with the above provided
Hackathon themes.
You can start working on your problem statement once you have finalized it.
Post-training can be done onsite on 25th & 26th when you receive compute
credits for HuggingFace.
Before the onsite, we suggest you work on building the environment, agent
behaviours, reward model and evaluate if your work aligns with the judging
criteria given below.

## Technical stack
- OpenEnv (latest) for env spec (FastAPI server)
- Pydantic models: Observation, Action, Reward
- Python astar.py for verifiable reward (no LLM judge)
- Qwen2.5-0.5B-Instruct with Unsloth QLoRA (4-bit, r=32)
- TRL GRPOTrainer for training
- HuggingFace Spaces for deployment
- Pygame for level rendering → PNG → GIF

## Grid format
8 rows x 16 cols. Tiles: . (empty) # (wall) ^ (spike) $ (coin) E (enemy) P (player start) G (goal)

## Three tasks
1. first_steps (easy): floor pre-built, add 2-4 coins, 0-2 spikes, path_len target=16
2. gap_jumper (medium): floor has gaps, add platforms + coins + 1 enemy, path_len target=22  
3. symmetric_shrine (hard): empty grid, design from scratch, symmetric, path_len target=30

## NPC personality system (expansion)
Each episode has a personality: brave / cautious / explorer
- brave: reward high enemy count + narrow paths
- cautious: reward wide paths + few spikes
- explorer: reward multiple branching routes

#Expansion 1 — Different Game Types (Same Environment, Different Modes)
Instead of only platformer levels, the AI can learn to design for:
ModeWhat AI DesignsHow It's ScoredPlatformer (current)Jump-and-run level with coins, spikes, wallsA* pathfinding + difficulty bandDungeon / RPG roomA room with enemies, treasure, a locked doorCan player reach key → then door?Puzzle roomPush blocks to open a pathSolver algorithm checks if solvableChase sequenceNarrow corridors, enemy chasing playerTension score — how many near-death moments

Expansion 2 — NPC Personalities (This Is The Wow Factor)
This is your best idea and nobody has done this. Here's what it means in simple terms:
Right now the AI designs a level for a generic player. But what if the level is designed for a specific character with a specific personality?
Example:

Brave player — wants risky levels, many enemies, narrow paths, high reward
Cautious player — wants wider paths, fewer spikes, more checkpoints
Explorer player — wants hidden rooms, secret coins, multiple paths
Speedrunner — wants straight direct path, minimal obstacles, fast completion

The AI gets told which player personality it's designing for, and it learns to make different levels for each type. The reward function checks if the level actually matches the personality.
How to score this:

Brave personality → reward levels with high enemy count + narrow paths
Cautious → reward wide open paths + few enemies
Explorer → reward multiple branching paths

## Reward function (NO LLM calls — pure Python)
R = 2.0 * solvable (A* hard gate)
  + 1.0 * difficulty_band (gaussian at target path_len)
  + 0.5 * coin_reachable_fraction
  + 0.5 * symmetry_iou
  + 0.5 * personality_match
  + 0.25 * xml_format_count
  - 0.5 * if trivially empty
  - 0.3 * if repeated tile spam

add more reward function sutable accoridng to rules and judgeing criteria  and both expansions.


## Judging criteria (weighted)
Judging Criteria
Minimum requirements:
●
Usage of OpenEnv (latest release)
●
Show a minimal training script for your environment using Unsloth or HF TRL in
●
●
Colab
Write a mini-blog on HuggingFace or mini-video on YouTube talking about your
submission, <2 minutes
Your OpenEnv compliant environment should be hosted on Hugging Face
Spaces.
Judging Overview
●
Evaluation: Teams will be scored based on the following criteria:
1. Environment Innovation (40%): Is the environment novel, creative, or
challenging? Does it meaningfully test the agent’s behavior?
2. Storytelling (30%): Does the team clearly explain the problem,
environment, and agent behavior? Is the demo engaging and easy to
follow?
3. Showing Improvement in Rewards (20%): Does the demo provide
observable evidence of training progress (reward curves, metrics, or
before/after behavior)?
4. Reward and Training Script/Pipeline Setup (10%): Is the reward logic
coherent, and does the pipeline produce meaningful improvement in
the agent’s inference (how it acts in the environment)?
OpenEnv Hackathon - What Judges Look For
This guide tells you what makes a strong submission for the OpenEnv Hackathon
(India 2026).
Read it before you start building, and again before you submit.
For the list of themes and example problems, refer to the top sections.
NOTE: Please remember only one submission per team. If you have multiple ideas, pick
the best one and go for it. Please make sure that the URL link of your environment is
submitted as judges will pull the environment from the URL to evaluate it. Changes or
commits after the submission deadline will not be considered.
TL;DR
Build an environment that an LLM could actually be trained on to get measurably
better at
something interesting. Then show that training. Then tell the story.
A messy but ambitious environment with real training evidence beats a polished but
boring one.
Pick a problem that excites you (that energy comes through in the pitch).
Judging Criteria
Criterion: Environment Innovation
Weight: 40%
What it means:
Is the environment novel, creative, or genuinely challenging?
Does it meaningfully test agent behavior in a way that hasn't been done before?
Criterion: Storytelling & Presentation
Weight: 30%
What it means:
Can you clearly explain the problem, the environment, and what the agent learned?
Is the demo engaging and easy to follow for a non-technical audience?
Criterion: Showing Improvement in Rewards
Weight: 20%
What it means:
Is there observable evidence of training progress? Reward curves, before/after
behavior,
comparison against a baseline -- anything that proves the agent learned something.
Criterion: Reward & Training Pipeline
Weight: 10%
What it means:
Is the reward logic coherent? Does the pipeline produce meaningful improvement in
the trained
agent's behavior?
Minimum Submission Requirements
NOTE: These are non-negotiable. Submissions missing any of these are at a serious
disadvantage.
wheel.
Use OpenEnv (latest release). Build on top of the framework; don’t reinvent the
A working training script using Unsloth or Hugging Face TRL, ideally as a
Colab notebook so judges can re-run it.
Evidence that you actually trained; at minimum, loss and reward plots from a
real run.
A short writeup: a mini-blog on Hugging Face or a < 2 minute video on YouTube
explaining what your environment does and what you trained, or a short slide
deck of presentation. Please make sure that all materials are linked from your
README file so that judges can access them easily.
Push your environment to a Hugging Face Space so it’s discoverable and
runnable.
A README that motivates the problem, explains how the env works, and
shows results.
README should have a link to the environment in the Hugging Face
Space. It should also have all additional references to other materials (e.g.
videos, blog posts, slides, presentations, etc.) that you want to include.
Please do not include big video files in your Env submission on HF Hub as we
would like to have a small size for each env (Please use url as reference link to
additional materials).
What Makes a Submission Stand Out
Pick an ambitious, original problem
The themes (problems) are deliberately open. Use them as launching pads, not boxes.
Judges have seen a lot of chess, snake, tic-tac-toe, and grid-world clones. To score well
on innovation,
you need a genuinely fresh angle. Some questions to ask yourself:
●
Does this environment exist to teach an LLM something it currently can’t do
well?
●
Is the domain underexplored in RL/LLM training?
●
Could a researcher write a paper about training on this?
Design a reward signal that actually teaches
A great environment has a reward function that:
●
Provides a rich, informative signal (not just 0/1 at the end)
●
Captures something hard to measure in a clever way
●
Uses OpenEnv’s Rubric system thoughtfully (composable rubrics > monolithic
scoring)
●
Is hard to game; an agent that exploits the reward without solving the task
should not get high scores
Show real training, end to end
The bar isn’t “training script exists.
” The bar is “training script runs against the
environment, the
agent learns, and you can show it.
” Concretely:
●
Your training loop should connect to your environment (not a static dataset)
●
Train long enough that the curves mean something
●
Compare a trained agent vs. a random/untrained baseline; quantitative and/or
qualitative
●
Include the plots and numbers in your README and writeup
Make your plots readable
Reviewers spend seconds, not minutes, on each plot. Help them out:
●
●
●
Label both axes (e.g.
“training step” / “episode” on x,
“reward” / “loss” on y) and
include units where they apply
Save plots as .png or .jpg and commit them to the repo (don’t leave them only
in a Colab cell or a deleted Wandb run) (if you ran via Wandb, please include the
link to that specific run of your plots)
Embed the key plots in your README with a one-line caption explaining what
each one shows If you have multiple runs (baseline vs. trained, ablations, etc.),
put them on the same axes so the comparison is obvious
Tell a story, not an API doc
Your README, blog, and pitch should answer:
1. Problem) what capability gap or interesting domain are you targeting?
2. Environment) what does the agent see, do, and get rewarded for?
3. Results) what changed after training? Show it.
4. Why does it matter) who would care, and why?
A reviewer should be able to read your README in 3~5 minutes and want to try your
environment.
NOTE: If you have a video, HF post, or anything else interesting, please make sure that
it’s linked
from your README as a link.
Engineer it cleanly (table stakes)
Engineering quality matters less than ambition, but sloppy work hurts. Make sure you:
●
Use OpenEnv’s Environment / MCPEnvironment base classes properly
●
Respect the client / server separation (clients should never import server
internals)
●
Follow the standard Gym-style API (reset, step, state)
●
Have a valid openenv.yaml manifest
●
Don’t use reserved tool names (reset, step, state, close) for MCP tools
Final Note
Judges are looking for environments that push the frontier of what we can train LLMs
to do. Be
ambitious. Pick a problem you find genuinely interesting; that almost always produces
better
work than chasing what you think judges want. Good luck.

## Training Configuration

### Model
- Name: unsloth/Qwen2.5-0.5B-Instruct
- Loading: QLoRA, load_in_4bit=True
- LoRA rank: r=32, lora_alpha=32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- use_gradient_checkpointing: "unsloth"

### GRPOConfig settings (validated for T4 15GB)
- learning_rate: 5e-6
- optim: paged_adamw_8bit
- per_device_train_batch_size: 1
- gradient_accumulation_steps: 1
- num_generations: 4  ← drop to 2 if OOM
- max_prompt_length: 256
- max_completion_length: 768
- max_steps: 200  ← test with 50 first
- save_steps: 50  ← always keep this, allows resume if Colab disconnects
- beta: 0.04
- loss_type: dr_grpo
- mask_truncated_completions: True
- report_to: none
- output_dir: cubtrail_out

### Reward functions order in GRPOTrainer
1. format_reward  ← dense signal, model learns this first (XML tags)
2. env_reward     ← calls HF Space /step endpoint, returns reward.total

### Training dataset
- 200 prompts total
- Each prompt = one scenario from scenarios.py (task + personality combo)
- System prompt includes: personality awareness, JSON edit format, XML think tags
- Repeat scenarios if needed to reach 200

### HF Space URL (update this before training)
ENV_URL = "https://YOUR_USERNAME-tigercraft-env.hf.space"

### Expected training behavior
- Steps 0-30: model learns XML format first (format_reward rises)
- Steps 30-100: model starts making solvable levels (solvable_reward rises)
- Steps 100-200: model learns personality differences (personality_match rises)
- If reward completely flat after 50 steps → tell claude.ai immediately

### Checkpointing
- Checkpoints saved every 50 steps to cubtrail_out/
- If Colab disconnects: reload from latest checkpoint and continue
- Never restart from step 0 if checkpoint exists

### Resume training from checkpoint (if disconnected)
trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(resume_from_checkpoint="cubtrail_out/checkpoint-100"),
    ...
)

### Plots required for submission
1. reward_curve.png — x: training step, y: reward/mean. Both axes labeled.
2. personality_comparison.png — 3 columns (brave/cautious/explorer), 3 rows (step 0/100/200)
3. evolution.gif — 5 frames showing level quality at steps 0,50,100,150,200
All saved to repo root. Embedded in README with captions.

### Mini-blog outline (HuggingFace post)
Paragraph 1: Problem — level design is hard, takes weeks
Paragraph 2: Environment — TigerCraft, 3 tasks, 3 personalities, A* reward
Paragraph 3: Results — reward curve, before/after level images
Paragraph 4: Why it matters — first LLM PCG environment in OpenEnv
Link to: GitHub repo, HF Space, YouTube video, reward_curve.png


## File structure required
leveforge_env/
├── inference.py          (root, mandatory, OpenAI client, [START][STEP][END] logs)
├── openenv.yaml          (spec metadata with 3 tasks listed)
├── requirements.txt
├── README.md
├── models.py             (Pydantic: Observation, Action, Reward)
├── client.py             (CubTrailEnv client class)
├── server/
│   ├── app.py            (FastAPI with /reset /step /state)
│   ├── environment.py    (main env logic)
│   ├── tilemap.py        (grid + tile definitions)
│   ├── astar.py          (A* pathfinder, 4-connected, spike=death)
│   ├── renderer.py       (Pygame grid → PNG)
│   ├── rewards.py        (all 6 reward functions)
│   ├── scenarios.py      (pre-built scenarios for each task+personality)
│   ├── requirements.txt
│   └── Dockerfile
└── training/
    └── train_grpo.ipynb  (Colab notebook)