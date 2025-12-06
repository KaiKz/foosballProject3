#!/usr/bin/env python
"""
Evaluate trained SAC and TQC models on FoosballEnv.

Provides:
  - evaluate_model_vs_passive: protagonist vs passive black (no antagonist model)
  - evaluate_head_to_head: protagonist vs antagonist model (SAC vs TQC, TQC vs SAC)

Includes:
  - Hard step cap per episode to avoid hanging forever.
  - Simple device selection for evaluation (MPS / CUDA / CPU).
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from sb3_contrib import TQC

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Safety cap so an episode can never run forever during evaluation.
MAX_STEPS_PER_EPISODE = 500   # adjust if you want shorter/longer eval episodes


def get_eval_device():
    """
    Device selection for *evaluation* (more relaxed than training):

    1) MPS (Mac)
    2) CUDA
    3) CPU
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[EVAL] Using Apple MPS backend")
        return "mps"

    if torch.cuda.is_available():
        print("[EVAL] Using CUDA for evaluation")
        return "cuda"

    print("[EVAL] Using CPU for evaluation")
    return "cpu"


# ---------------------------------------------------------------------
# Env factory for evaluation
# ---------------------------------------------------------------------

def make_eval_env(antagonist_model=None, render_mode=None):
    """
    Create a single FoosballEnv for evaluation.

    - antagonist_model=None  -> passive / zero-control black side
    - antagonist_model=model -> that model controls black rods

    Arguments here mirror what you used in training, so behaviour is consistent.
    """
    env = FoosballEnv(
        antagonist_model=antagonist_model,
        render_mode=render_mode,
        verbose_mode=False,
        play_until_goal=True,
    )
    return env


# ---------------------------------------------------------------------
# 1) Protagonist vs passive black
# ---------------------------------------------------------------------

def evaluate_model_vs_passive(
    model,
    label: str,
    n_episodes: int = 10,
    render: bool = False,
):
    """
    Evaluate a single model (yellow/protagonist) vs passive black (no antagonist model).
    """

    env = make_eval_env(
        antagonist_model=None,   # passive black (same as your SAC training setup)
        render_mode="human" if render else None,
    )

    for ep_idx in range(n_episodes):
        print(f"[{label}] Starting episode {ep_idx + 1}/{n_episodes}...", flush=True)

        obs, _ = env.reset()
        done = False
        step_count = 0
        ep_return = 0.0
        goal_scored = False
        winning_goal = False
        info = {}

        while (not done) and (step_count < MAX_STEPS_PER_EPISODE):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            step_count += 1
            done = bool(terminated or truncated)

            if info.get("goal_scored", False):
                goal_scored = True
                winning_goal = info.get("winning_goal", False)

            if render:
                env.render()

        if not done and step_count >= MAX_STEPS_PER_EPISODE:
            print(
                f"[{label}] Episode {ep_idx + 1:03d} hit "
                f"MAX_STEPS_PER_EPISODE={MAX_STEPS_PER_EPISODE} without env.done=True. "
                f"Forcing truncation.",
                flush=True,
            )

        print(
            f"[{label}] Episode {ep_idx + 1:03d}: "
            f"return={ep_return:.1f}, steps={step_count}, "
            f"goal_scored={goal_scored}, winning_goal={winning_goal}",
            flush=True,
        )

    env.close()


# ---------------------------------------------------------------------
# 2) Head-to-head: protagonist vs antagonist model
# ---------------------------------------------------------------------

def evaluate_head_to_head(
    protagonist_model,
    antagonist_model,
    label: str,
    n_episodes: int = 10,
    render: bool = False,
):
    """
    Evaluate protagonist_model (yellow) vs antagonist_model (black).
    Uses same FoosballEnv as training, but with antagonist_model plugged in.
    """

    env = make_eval_env(
        antagonist_model=antagonist_model,
        render_mode="human" if render else None,
    )

    wins = 0
    losses = 0
    goals = 0
    total_return = 0.0
    total_steps = 0

    for ep_idx in range(n_episodes):
        print(f"[{label}] Starting episode {ep_idx + 1}/{n_episodes}...", flush=True)

        obs, _ = env.reset()
        done = False
        step_count = 0
        ep_return = 0.0
        info = {}

        while (not done) and (step_count < MAX_STEPS_PER_EPISODE):
            if render:
                env.render()

            action, _ = protagonist_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            step_count += 1
            done = bool(terminated or truncated)

        if not done and step_count >= MAX_STEPS_PER_EPISODE:
            print(
                f"[{label}] Episode {ep_idx + 1:03d} hit "
                f"MAX_STEPS_PER_EPISODE={MAX_STEPS_PER_EPISODE} without env.done=True. "
                f"Forcing truncation.",
                flush=True,
            )

        total_return += ep_return
        total_steps += step_count

        winning_goal = info.get("winning_goal", False)
        losing_goal = info.get("losing_goal", False)
        goal_scored = info.get("goal_scored", False)

        if winning_goal:
            wins += 1
        if losing_goal:
            losses += 1
        if goal_scored:
            goals += 1

        print(
            f"[{label}] Episode {ep_idx + 1:03d}: "
            f"return={ep_return:.1f}, steps={step_count}, "
            f"goal_scored={goal_scored}, "
            f"winning_goal={winning_goal}, losing_goal={losing_goal}",
            flush=True,
        )

    env.close()

    print(f"\n[{label}] Head-to-head summary:")
    print(f"  Episodes:          {n_episodes}")
    print(f"  Protagonist wins:  {wins}")
    print(f"  Protagonist losses:{losses}")
    print(f"  Total goals (any): {goals}")
    print(f"  Win rate:          {wins / n_episodes:.3f}")
    print(f"  Loss rate:         {losses / n_episodes:.3f}")
    print(f"  Avg return:        {total_return / n_episodes:.1f}")
    print(f"  Avg ep length:     {total_steps / n_episodes:.1f} steps\n")


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    device = get_eval_device()

    # Load trained models (paths match your training script)
    sac_model = SAC.load("foosball_sac_nenv8_model.zip", device=device)
    tqc_model = TQC.load("foosball_tqc_nenv8_model.zip", device=device)

    try:
        # # 1) Each vs passive black (no antagonist model)
        # evaluate_model_vs_passive(
        #     sac_model,
        #     "SAC vs passive black",
        #     n_episodes=50,
        #     render=True,
        # )
        # evaluate_model_vs_passive(
        #     tqc_model,
        #     "TQC vs passive black",
        #     n_episodes=50,
        #     render=True,
        # )

        # 2) Head-to-head: SAC (yellow) vs TQC (black)
        evaluate_head_to_head(
            protagonist_model=sac_model,
            antagonist_model=tqc_model,
            label="SAC (yellow) vs TQC (black)",
            n_episodes=50,
            render=True,      # flip to True if you want to watch a few
        )

        # # 3) Head-to-head: TQC (yellow) vs SAC (black)
        # evaluate_head_to_head(
        #     protagonist_model=tqc_model,
        #     antagonist_model=sac_model,
        #     label="TQC (yellow) vs SAC (black)",
        #     n_episodes=50,
        #     render=True,
        # )

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation stopped by user.", flush=True)
