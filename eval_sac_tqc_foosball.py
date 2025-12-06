#!/usr/bin/env python
"""
Evaluate trained SAC and TQC models on FoosballEnv.

Provides:
  - evaluate_model_vs_passive: protagonist vs passive-lifted black
  - evaluate_head_to_head: protagonist vs antagonist model (SAC vs TQC, TQC vs SAC)
Can optionally show viewer for a small number of episodes.
"""

import numpy as np
from stable_baselines3 import SAC
from sb3_contrib import TQC

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def evaluate_model_vs_passive(model, label: str, n_episodes: int = 50, render: bool = False):
    """
    Evaluate a single model as protagonist (yellow) vs passive-lifted black.

    Uses FoosballEnv(antagonist_model=None), so the 'lift when passive' logic
    from FoosballEnv.step is in effect.
    """

    env = FoosballEnv(
        antagonist_model=None,
        render_mode="human" if render else None,
        verbose_mode=False,
        play_until_goal=True,
    )

    wins = 0
    goals = 0
    total_return = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_steps = 0

        while not done:
            if render:
                env.render()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_ret += float(reward)
            ep_steps += 1

        total_return += ep_ret
        total_steps += ep_steps

        if info.get("winning_goal", False):
            wins += 1
        if info.get("goal_scored", False):
            goals += 1

        print(
            f"[{label}] Episode {ep+1:03d}: "
            f"return={ep_ret:.1f}, steps={ep_steps}, "
            f"goal_scored={info.get('goal_scored', False)}, "
            f"winning_goal={info.get('winning_goal', False)}"
        )

    env.close()

    print(f"\n[{label}] Summary vs passive black:")
    print(f"  Episodes:      {n_episodes}")
    print(f"  Wins:          {wins}")
    print(f"  Goals (any):   {goals}")
    print(f"  Win rate:      {wins / n_episodes:.3f}")
    print(f"  Avg return:    {total_return / n_episodes:.1f}")
    print(f"  Avg ep length: {total_steps / n_episodes:.1f} steps\n")


def evaluate_head_to_head(
    protagonist_model,
    antagonist_model,
    label: str,
    n_episodes: int = 50,
    render: bool = False,
):
    """
    Evaluate protagonist_model (yellow) vs antagonist_model (black).
    """

    env = FoosballEnv(
        antagonist_model=antagonist_model,
        render_mode="human" if render else None,
        verbose_mode=False,
        play_until_goal=True,
    )

    wins = 0
    losses = 0
    goals = 0
    total_return = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_steps = 0

        while not done:
            if render:
                env.render()

            action, _ = protagonist_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_ret += float(reward)
            ep_steps += 1

        total_return += ep_ret
        total_steps += ep_steps

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
            f"[{label}] Episode {ep+1:03d}: "
            f"return={ep_ret:.1f}, steps={ep_steps}, "
            f"goal_scored={goal_scored}, "
            f"winning_goal={winning_goal}, losing_goal={losing_goal}"
        )

    env.close()

    print(f"\n[{label}] Head-to-head summary:")
    print(f"  Episodes:        {n_episodes}")
    print(f"  Protagonist wins:{wins}")
    print(f"  Protagonist losses:{losses}")
    print(f"  Total goals (any):{goals}")
    print(f"  Win rate:        {wins / n_episodes:.3f}")
    print(f"  Loss rate:       {losses / n_episodes:.3f}")
    print(f"  Avg return:      {total_return / n_episodes:.1f}")
    print(f"  Avg ep length:   {total_steps / n_episodes:.1f} steps\n")


if __name__ == "__main__":
    # Load trained models (saved by train_sac_tqc_foosball.py)
    # Force device="cpu" for evaluation to avoid GPU/MPS issues.
    sac_model = SAC.load("foosball_sac_nenv8_model.zip", device="cpu")
    tqc_model = TQC.load("foosball_tqc_nenv8_model.zip", device="cpu")

    # 1) Each vs passive-lifted black
    evaluate_model_vs_passive(sac_model, "SAC vs passive black", n_episodes=50, render=False)
    evaluate_model_vs_passive(tqc_model, "TQC vs passive black", n_episodes=50, render=False)

    # 2) Head-to-head
    evaluate_head_to_head(
        protagonist_model=sac_model,
        antagonist_model=tqc_model,
        label="SAC (yellow) vs TQC (black)",
        n_episodes=50,
        render=False,
    )

    evaluate_head_to_head(
        protagonist_model=tqc_model,
        antagonist_model=sac_model,
        label="TQC (yellow) vs SAC (black)",
        n_episodes=50,
        render=False,
    )

    # If you want to WATCH a few games, try e.g.:
    # evaluate_head_to_head(sac_model, tqc_model, "SAC vs TQC (viewer)", n_episodes=5, render=True)
