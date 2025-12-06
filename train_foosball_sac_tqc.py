#!/usr/bin/env python
"""
Train SAC/TQC on FoosballEnv and plot:
- goal rate per episode
- mean return
- episode length

Device policy:
- Try CUDA first
- Then MPS
- If neither is available, RAISE and exit (never use CPU)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import gymnasium as gym
from stable_baselines3 import SAC
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


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
# Env factory
# =========================================================

def make_env(seed: int = 0):
    """
    Create a monitored FoosballEnv for SB3.
    """
    def _init():
        env = FoosballEnv(
            render_mode=None,       # no viewer during training
            verbose_mode=False,
            play_until_goal=False,  # use your termination logic
        )
        env = Monitor(env)         # adds episode stats into info["episode"]
        env.reset(seed=seed)
        return env
    return _init


# =========================================================
# Callback to collect per-episode stats
# =========================================================

class EpisodeStatsCallback(BaseCallback):
    """
    Collect:
      - episode return
      - episode length
      - goal_scored flag from env info
    and keep them in lists so we can plot later.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_goals = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if done:
                ep_info = info.get("episode")
                if ep_info is not None:
                    self.episode_returns.append(float(ep_info["r"]))
                    self.episode_lengths.append(int(ep_info["l"]))
                    self.episode_goals.append(
                        float(info.get("goal_scored", False))
                    )

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


# =========================================================
# Main training routine
# =========================================================

def train(
    algo: str = "sac",
    total_timesteps: int = 200_000,
    seed: int = 0,
):
    """
    algo: "sac" or "tqc"
    """
    # Decide device (CUDA -> MPS -> error if neither)
    device = get_training_device()

    vec_env = DummyVecEnv([make_env(seed=seed)])

    if algo.lower() == "sac":
        AlgoClass = SAC
        algo_name = "SAC"
    elif algo.lower() == "tqc":
        AlgoClass = TQC
        algo_name = "TQC"
    else:
        raise ValueError("algo must be 'sac' or 'tqc'")

    callback = EpisodeStatsCallback(verbose=1)

    print(f"[TRAIN] Starting {algo_name} training for {total_timesteps} timesteps on device={device}")

    model = AlgoClass(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,  # <<< enforce GPU device here
        tensorboard_log=os.path.join("tb_logs", "foosball_" + algo_name.lower()),
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.01,
        train_freq=1,
        gradient_steps=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    save_path = f"foosball_{algo_name.lower()}_model"
    model.save(save_path)
    print(f"[TRAIN] Saved {algo_name} model to {save_path}.zip")

    plot_training_stats(callback, algo_name=algo_name)

    return model, callback


if __name__ == "__main__":
    # Example: SAC only. You can uncomment TQC run below if you want both.
    model_sac, cb_sac = train(
        algo="sac",
        total_timesteps=100_000,
        seed=0,
    )

    # model_tqc, cb_tqc = train(
    #     algo="tqc",
    #     total_timesteps=100_000,
    #     seed=1,
    # )
