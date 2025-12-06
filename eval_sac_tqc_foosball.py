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
import os
import matplotlib
matplotlib.use("Agg")           # non-interactive backend, safe on servers
import matplotlib.pyplot as plt
# import times
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import csv
import numpy as np


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Safety cap so an episode can never run forever during evaluation.
MAX_STEPS_PER_EPISODE = 3000   # adjust if you want shorter/longer eval episodes
render=False,


class RandomPolicy:
    """
    Simple baseline: ignores observations and samples uniformly
    from the env's action_space.
    """
    def __init__(self):
        # Create a temporary env just to grab the correct action_space
        tmp_env = make_eval_env(antagonist_model=None, render_mode=None)
        self.action_space = tmp_env.action_space
        tmp_env.close()

    def predict(self, obs, deterministic=True):
        # SB3-like API: returns (action, state)
        action = self.action_space.sample()
        return action, None


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
        render_mode="human" if render else None,
        verbose_mode=False,
        play_until_goal=False,
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

    Returns a dict with raw per-episode stats so we can plot / log later.
    """

    env = make_eval_env(
        antagonist_model=None,   # passive black (same as your SAC training setup)
        render_mode="human" if render else None,
    )

    ep_returns = []
    ep_steps = []
    ep_goals = []

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
                # time.sleep(1/60.0) 

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

        ep_returns.append(ep_return)
        ep_steps.append(step_count)
        ep_goals.append(1.0 if goal_scored else 0.0)

    env.close()

    stats = {
        "label": label,
        "returns": ep_returns,
        "steps": ep_steps,
        "goals": ep_goals,
    }
    return stats



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

    Returns a dict of per-episode stats + win/loss counts.
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

    ep_returns = []
    ep_steps = []

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
                # time.sleep(1/60.0) 

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

        ep_returns.append(ep_return)
        ep_steps.append(step_count)

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

    stats = {
        "label": label,
        "returns": ep_returns,
        "steps": ep_steps,
        "wins": wins,
        "losses": losses,
        "goals": goals,
    }
    return stats


def _summarize_eval_dict(d):
    returns = np.array(d.get("returns", []), dtype=float)
    steps   = np.array(d.get("steps", []), dtype=float)
    goals   = np.array(d.get("goals", []), dtype=float)

    n_episodes = len(returns)

    # For SAC vs passive and TQC vs passive, there are no wins/losses.
    wins   = float(d.get("wins", 0))
    losses = float(d.get("losses", 0))

    summary = {
        "label": d.get("label", ""),
        "episodes": int(n_episodes),
        "mean_return": float(returns.mean()) if n_episodes > 0 else 0.0,
        "std_return": float(returns.std()) if n_episodes > 0 else 0.0,
        "mean_steps": float(steps.mean()) if n_episodes > 0 else 0.0,
        "win_rate": wins / n_episodes if (n_episodes > 0 and "wins" in d) else 0.0,
        "loss_rate": losses / n_episodes if (n_episodes > 0 and "losses" in d) else 0.0,
        "goal_rate": float(goals.mean()) if n_episodes > 0 else 0.0,
    }
    return summary




def save_eval_summary_and_plots(
    sac_vs_passive,
    tqc_vs_passive,
    sac_vs_tqc,
    out_dir: str = "foosball_plots",
):
    """
    Given raw stats from:
      - SAC vs passive
      - TQC vs passive
      - SAC (yellow) vs TQC (black)

    Save:
      - eval_summary.csv with basic stats
      - a few PNG plots comparing SAC and TQC
    """

    os.makedirs(out_dir, exist_ok=True)

    sac_passive_summary = _summarize_eval_dict(sac_vs_passive)
    tqc_passive_summary = _summarize_eval_dict(tqc_vs_passive)
    sac_tqc_summary = _summarize_eval_dict(sac_vs_tqc)

    # ---------------- CSV LOG ----------------
    csv_path = os.path.join(out_dir, "eval_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "episodes",
                "mean_return",
                "std_return",
                "mean_steps",
                "win_rate",
                "loss_rate",
                "goal_rate",
            ],
        )
        writer.writeheader()
        for row in [sac_passive_summary, tqc_passive_summary, sac_tqc_summary]:
            writer.writerow(row)

    print(f"[EVAL] Saved evaluation summary CSV to {csv_path}")

    # ---------------- PLOT 1: mean return vs passive ----------------
    x = np.arange(2)
    width = 0.6
    sac_val = sac_passive_summary["mean_return"]
    tqc_val = tqc_passive_summary["mean_return"]

    plt.figure()
    plt.bar(
        x,
        [sac_val, tqc_val],
        width=width,
        tick_label=["SAC (yellow)", "TQC (black)"],
        color=["gold", "black"],     # << yellow = SAC, black = TQC
    )
    plt.ylabel("Mean return vs passive")
    plt.title("Foosball: mean return vs passive opponent")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_return_vs_passive.png"))
    plt.close()

    # ---------------- PLOT 2: win rate in head-to-head ----------------
    plt.figure()
    plt.bar(
        ["SAC (yellow) vs TQC (black)"],
        [sac_tqc_summary["win_rate"]],
        color=["gold"],
    )
    plt.ylim(0, 1.0)
    plt.ylabel("Win rate (SAC as yellow)")
    plt.title("Head-to-head win rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_to_head_win_rate.png"))
    plt.close()

    # ---------------- PLOT 3: head-to-head return distribution ----------------
    plt.figure()
    plt.hist(
        sac_vs_tqc["returns"],
        bins=20,
        alpha=0.7,
        color="gold",
        label="SAC (yellow) returns",
    )
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.title("Head-to-head episode return distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_to_head_returns_hist.png"))
    plt.close()

    print(f"[EVAL] Saved evaluation plots to {out_dir}")

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    device = get_eval_device()

    # 0) Random baseline
    random_policy = RandomPolicy()
    random_vs_passive = evaluate_model_vs_passive(
        random_policy,
        "Random policy vs passive black",
        n_episodes=50,
        render=False,   # don't render for baseline
    )

    # Print quick stats for the baseline
    import numpy as np
    random_goal_rate = float(np.mean(random_vs_passive["goals"]))
    random_mean_return = float(np.mean(random_vs_passive["returns"]))
    print("\n[BASELINE] Random vs passive:")
    print(f"  goal_rate = {random_goal_rate:.3f}")
    print(f"  mean_return = {random_mean_return:.1f}\n")

    # 1) Load trained models
    sac_model = SAC.load("foosball_sac_nenv8_model.zip", device=device)
    tqc_model = TQC.load("foosball_tqc_nenv8_model.zip", device=device)

    try:
        # 2) SAC vs passive (no render – faster, and avoids viewer issues)
        sac_vs_passive = evaluate_model_vs_passive(
            sac_model,
            "SAC (yellow) vs passive black",
            n_episodes=50,
            render=False,
        )

        # 3) TQC vs passive
        tqc_vs_passive = evaluate_model_vs_passive(
            tqc_model,
            "TQC (black) vs passive black",
            n_episodes=50,
            render=False,
        )

        # 4) SAC vs TQC head-to-head (also no render for big batch)
        sac_vs_tqc = evaluate_head_to_head(
            protagonist_model=sac_model,   # yellow
            antagonist_model=tqc_model,    # black
            label="SAC (yellow) vs TQC (black)",
            n_episodes=50,
            render=False,
        )
        
                # 4) Comparison charts + CSV for discussion
        save_eval_summary_and_plots(
            sac_vs_passive=sac_vs_passive,
            tqc_vs_passive=tqc_vs_passive,
            sac_vs_tqc=sac_vs_tqc,
            out_dir="foosball_plots",
        )


        # 5) (Optional) small visual demo *afterwards* so viewer doesn’t explode
        evaluate_head_to_head(
            protagonist_model=sac_model,
            antagonist_model=tqc_model,
            label="SAC (yellow) vs TQC (black) [visual demo]",
            n_episodes=3,
            render=True,
        )

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation stopped by user.", flush=True)
