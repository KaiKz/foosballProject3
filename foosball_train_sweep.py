#!/usr/bin/env python
"""
Sweep over reward parameters for SAC on FoosballEnv.

For each RewardWeights config:
  - Train SAC vs passive-lifted black (no antagonist_model).
  - Collect episode returns.
  - Compute a simple score (mean return over last K episodes).
  - Plot all curves together for visual comparison.
  - Print a ranked summary of configs by score.

This script does *not* do the big fixed-parameter SAC+TQC training anymore.
"""

import os
import numpy as np

# --- IMPORTANT: use non-Tk backend to avoid Tcl_AsyncDelete crashes ---
import matplotlib
matplotlib.use("Agg")          # must come BEFORE importing pyplot
import matplotlib.pyplot as plt

import torch
import gymnasium as gym
from stable_baselines3 import SAC
from sb3_contrib import TQC  # (unused now, but you can keep or remove)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import (
    FoosballEnv,
    RewardWeights,
)

# =========================================================
# Device selection: CUDA -> MPS -> error
# =========================================================

def get_training_device():
    """
    Choose device strictly in this order:
    1) CUDA
    2) MPS
    If neither is available, raise an error (do NOT fall back to CPU).
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[DEVICE] Using CUDA GPU: {gpu_name}")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[DEVICE] Using Apple MPS backend")
        return "mps"

    # Explicitly refuse to train on CPU
    raise RuntimeError(
        "[DEVICE ERROR] No CUDA or MPS device available. "
        "Refusing to train on CPU. Make sure you are on a GPU machine with "
        "CUDA (or MPS on macOS) properly configured."
    )


# =========================================================
# Env factory (supports multi-env + optional antagonist)
# =========================================================

def make_env(seed: int = 0,
             antagonist_model=None,
             reward_weights: RewardWeights | None = None):
    def _init():
        env = FoosballEnv(
            antagonist_model=antagonist_model,
            render_mode=None,
            verbose_mode=False,
            play_until_goal=False,
            reward_weights=reward_weights,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# =========================================================
# Callback to collect per-episode stats
# =========================================================

class EpisodeStatsCallback(BaseCallback):
    """
    Collect per-episode stats AND log them to TensorBoard.

    Logs (per episode):
      - <prefix>/ep_return
      - <prefix>/ep_length
      - <prefix>/ep_goal

    Plus moving means over the last K episodes:
      - <prefix>/ep_return_mean_K
      - <prefix>/ep_length_mean_K
      - <prefix>/ep_goal_rate_K
    """

    def __init__(self, verbose: int = 1, tensorboard_prefix: str = "episode"):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_goals = []
        self.prefix = tensorboard_prefix
        self._n_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue

            ep_info = info.get("episode")
            if ep_info is None:
                continue

            r = float(ep_info["r"])
            l = int(ep_info["l"])
            g = float(info.get("goal_scored", False))

            self.episode_returns.append(r)
            self.episode_lengths.append(l)
            self.episode_goals.append(g)
            self._n_episodes += 1

            # --- log raw episode stats ---
            self.logger.record(f"{self.prefix}/ep_return", r)
            self.logger.record(f"{self.prefix}/ep_length", l)
            self.logger.record(f"{self.prefix}/ep_goal", g)

            # --- log moving averages over last K episodes ---
            K = 50
            recent_returns = self.episode_returns[-K:]
            recent_lengths = self.episode_lengths[-K:]
            recent_goals   = self.episode_goals[-K:]

            self.logger.record(
                f"{self.prefix}/ep_return_mean_{K}",
                float(np.mean(recent_returns)),
            )
            self.logger.record(
                f"{self.prefix}/ep_length_mean_{K}",
                float(np.mean(recent_lengths)),
            )
            self.logger.record(
                f"{self.prefix}/ep_goal_rate_{K}",
                float(np.mean(recent_goals)),
            )

            if self.verbose and (self._n_episodes % 50 == 0):
                print(f"[CB] Logged {self._n_episodes} episodes to TensorBoard")

        return True


# =========================================================
# Plot helpers
# =========================================================

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    if len(x) < window:
        return x.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")


def plot_training_stats(cb: EpisodeStatsCallback,
                        algo_name: str = "SAC",
                        out_dir: str = "foosball_plots",
                        window: int = 20):
    os.makedirs(out_dir, exist_ok=True)

    ep_returns = np.array(cb.episode_returns, dtype=np.float32)
    ep_lengths = np.array(cb.episode_lengths, dtype=np.float32)
    ep_goals   = np.array(cb.episode_goals, dtype=np.float32)

    episodes = np.arange(1, len(ep_returns) + 1)

    ret_ma   = moving_average(ep_returns, window)
    len_ma   = moving_average(ep_lengths, window)
    goal_ma  = moving_average(ep_goals, window)

    if len(ep_returns) >= window:
        ma_episodes = episodes[window - 1:]
    else:
        ma_episodes = episodes

    # ---- Returns ----
    plt.figure()
    plt.plot(episodes, ep_returns, alpha=0.3, label="Episode return")
    plt.plot(ma_episodes, ret_ma, linewidth=2.0, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{algo_name} on FoosballEnv – Episode Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{algo_name.lower()}_returns.png"))
    plt.close()

    # ---- Episode length ----
    plt.figure()
    plt.plot(episodes, ep_lengths, alpha=0.3, label="Episode length")
    plt.plot(ma_episodes, len_ma, linewidth=2.0, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title(f"{algo_name} on FoosballEnv – Episode Length")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{algo_name.lower()}_lengths.png"))
    plt.close()

    # ---- Goal rate ----
    plt.figure()
    plt.plot(episodes, ep_goals, "o", alpha=0.2, label="Goal (1) / no goal (0)")
    plt.plot(ma_episodes, goal_ma, linewidth=2.0, label=f"{window}-ep goal rate")
    plt.xlabel("Episode")
    plt.ylabel("Goal rate")
    plt.ylim(-0.05, 1.05)
    plt.title(f"{algo_name} on FoosballEnv – Goal Rate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{algo_name.lower()}_goals.png"))
    plt.close()

    print(f"[PLOTS] Saved plots to: {out_dir}")


def plot_reward_sweep_comparison(callbacks, labels,
                                 out_dir: str = "foosball_plots",
                                 window: int = 50):
    """
    Compare several reward-weight settings on the same plot by
    overlaying their moving-average episode returns.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    for cb, label in zip(callbacks, labels):
        ep_returns = np.array(cb.episode_returns, dtype=np.float32)
        if len(ep_returns) == 0:
            continue

        episodes = np.arange(1, len(ep_returns) + 1)
        ret_ma = moving_average(ep_returns, window)

        if len(ep_returns) >= window:
            ma_episodes = episodes[window - 1:]
        else:
            ma_episodes = episodes

        plt.plot(ma_episodes, ret_ma, linewidth=2.0, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"{window}-episode mean return")
    plt.title("Reward-weight sweep: SAC episode returns")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_sweep_returns.png"))
    plt.close()

    print(f"[PLOTS] Saved reward sweep comparison to {out_dir}/reward_sweep_returns.png")


# =========================================================
# Main training routine (multi-env + optional antagonist)
# =========================================================

def train(
    algo: str = "sac",
    total_timesteps: int = 2_000_000,
    seed: int = 0,
    n_envs: int = 8,
    antagonist_model=None,
    reward_weights: RewardWeights | None = None,
    run_name: str | None = None,
):
    device = get_training_device()

    if antagonist_model is None:
        vec_env = SubprocVecEnv(
            [make_env(seed=seed + i,
                      antagonist_model=None,
                      reward_weights=reward_weights)
             for i in range(n_envs)]
        )
    else:
        vec_env = DummyVecEnv(
            [make_env(seed=seed + i,
                      antagonist_model=antagonist_model,
                      reward_weights=reward_weights)
             for i in range(n_envs)]
        )

    if algo.lower() == "sac":
        AlgoClass = SAC
        algo_name = "SAC"
    elif algo.lower() == "tqc":
        AlgoClass = TQC
        algo_name = "TQC"
    else:
        raise ValueError("algo must be 'sac' or 'tqc'")

    callback = EpisodeStatsCallback(
        verbose=1,
        tensorboard_prefix=f"{algo_name}/episode",
    )

    log_dir = os.path.join("tb_logs", f"foosball_{algo_name.lower()}_nenv{n_envs}")
    os.makedirs(log_dir, exist_ok=True)

    print(
        f"[TRAIN] Starting {algo_name} training for {total_timesteps} timesteps "
        f"on device={device} with n_envs={n_envs}, "
        f"antagonist={'None' if antagonist_model is None else 'provided'}, "
        f"reward_weights={reward_weights}"
    )
    tb_name = run_name or f"{algo_name}_nenv{n_envs}"
    model = AlgoClass(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        train_freq=16,
        gradient_steps=32,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=tb_name,
    )

    save_path = f"foosball_{algo_name.lower()}_{tb_name}"
    model.save(save_path)
    print(f"[TRAIN] Saved {algo_name} model to {save_path}.zip")

    # still save per-config plots if you want
    plot_training_stats(callback, algo_name=algo_name)

    return model, callback


# =========================================================
# Reward sweep (this is the only thing run in __main__)
# =========================================================

def run_simple_reward_search():
    """
    Automatic search over reward parameters.

    Trains SAC several times with different RewardWeights, collects
    callbacks, and then:
      - makes a comparison plot using plot_reward_sweep_comparison(...)
      - prints a ranked list of configs by score
    """
    # You can tweak these configs; keep it small so it actually finishes.
    reward_configs = [
        RewardWeights(
            progress_w=1.0,
            distance_w=0.3,
            speed_w=0.0,
            control_cost_w=0.001,
            time_penalty=-0.1,
        ),
        RewardWeights(
            progress_w=2.0,
            distance_w=0.5,
            speed_w=0.01,
            control_cost_w=0.001,
            time_penalty=-0.1,
        ),
        RewardWeights(
            progress_w=3.0,
            distance_w=0.5,
            speed_w=0.02,
            control_cost_w=0.001,
            time_penalty=-0.1,
        ),
    ]

    callbacks = []
    labels = []
    scores = []

    for i, rw in enumerate(reward_configs):
        label = f"SAC_rw{i}_p{rw.progress_w}_d{rw.distance_w}"
        print(f"\n[SWEEP] Training config {i}: {label} -> {rw}")

        model, cb = train(
            algo="sac",
            total_timesteps=300_000,     # shorter than your “main” run
            seed=100 + i,
            n_envs=8,
            antagonist_model=None,
            reward_weights=rw,
            run_name=label,
        )

        callbacks.append(cb)
        labels.append(label)

        # simple score: mean of last 100 episode returns
        if len(cb.episode_returns) > 0:
            K = min(100, len(cb.episode_returns))
            score = float(np.mean(cb.episode_returns[-K:]))
        else:
            score = -np.inf
        scores.append(score)
        print(f"[SWEEP] Config {label} score (last {K} eps mean) = {score:.2f}")

    # Comparison chart (all configs on one plot)
    plot_reward_sweep_comparison(callbacks, labels)

    # Rank configs by score
    ranking = sorted(
        zip(labels, reward_configs, scores),
        key=lambda t: t[2],
        reverse=True,
    )

    print("\n========== REWARD SWEEP RANKING (best first) ==========")
    for rank, (label, rw, score) in enumerate(ranking, start=1):
        print(
            f"{rank:2d}. {label}: score={score:.2f}, "
            f"progress_w={rw.progress_w}, distance_w={rw.distance_w}, "
            f"speed_w={rw.speed_w}, time_penalty={rw.time_penalty}"
        )


if __name__ == "__main__":
    # This script now ONLY runs the reward-parameter sweep.
    run_simple_reward_search()
