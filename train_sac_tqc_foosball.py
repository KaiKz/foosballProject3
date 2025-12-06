#!/usr/bin/env python
"""
Train SAC and then TQC on FoosballEnv:

1) Train SAC as protagonist vs passive-lifted black (no antagonist_model).
2) Train TQC as protagonist vs frozen SAC antagonist.

For both:
- Use multi-env training.
- Log episode returns, lengths, and goal rate.
- Save plots to foosball_plots/.
- Save models as:
    foosball_sac_nenv<N>_model.zip
    foosball_tqc_nenv<N>_model.zip
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
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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
# Env factory (supports multi-env + optional antagonist)
# =========================================================

def make_env(seed: int = 0, antagonist_model=None):
    """
    Create a monitored FoosballEnv for SB3.

    antagonist_model:
        If not None, this model will control the black rods (antagonist side)
        via env.antagonist_model. It is treated as FROZEN (no updates).
    """

    def _init():
        env = FoosballEnv(
            antagonist_model=antagonist_model,
            render_mode=None,       # no viewer during training
            verbose_mode=False,     # IMPORTANT: keep False for speed
            play_until_goal=True,   # let env termination logic handle goals / timeouts
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
# Main training routine (multi-env + optional antagonist)
# =========================================================

def train(
    algo: str = "sac",
    total_timesteps: int = 2_000_000,
    seed: int = 0,
    n_envs: int = 8,
    antagonist_model=None,
):
    """
    algo: "sac" or "tqc"
    total_timesteps: counted across all envs (SB3 convention).
    n_envs: number of parallel envs.
    antagonist_model:
        If provided, this model will control the black rods (env.antagonist_model)
        during training. It is treated as FROZEN (no updates).
    """

    device = get_training_device()

    # VecEnv choice:
    # - If antagonist_model is None -> SubprocVecEnv for speed (no pickling of models).
    # - If antagonist_model is not None -> DummyVecEnv (we close over a live SB3 model).
    if antagonist_model is None:
        vec_env = SubprocVecEnv(
            [make_env(seed=seed + i, antagonist_model=None) for i in range(n_envs)]
        )
    else:
        vec_env = DummyVecEnv(
            [make_env(seed=seed + i, antagonist_model=antagonist_model) for i in range(n_envs)]
        )

    if algo.lower() == "sac":
        AlgoClass = SAC
        algo_name = "SAC"
    elif algo.lower() == "tqc":
        AlgoClass = TQC
        algo_name = "TQC"
    else:
        raise ValueError("algo must be 'sac' or 'tqc'")

    callback = EpisodeStatsCallback(verbose=1)

    log_dir = os.path.join("tb_logs", f"foosball_{algo_name.lower()}_nenv{n_envs}")
    os.makedirs(log_dir, exist_ok=True)

    print(
        f"[TRAIN] Starting {algo_name} training for {total_timesteps} timesteps "
        f"on device={device} with n_envs={n_envs}, "
        f"antagonist={'None' if antagonist_model is None else 'provided'}"
    )

    model = AlgoClass(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=1024,
        gamma=0.99,
        tau=0.01,
        train_freq=1,          # 1 step per env => n_envs transitions per update
        gradient_steps=4,      # multiple gradient updates per collection
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    save_path = f"foosball_{algo_name.lower()}_nenv{n_envs}_model"
    model.save(save_path)
    print(f"[TRAIN] Saved {algo_name} model to {save_path}.zip")

    plot_training_stats(callback, algo_name=algo_name)

    return model, callback

if __name__ == "__main__":
    # ------------------- PHASE 1: Train SAC vs passive-lifted black -------------------
    TRAIN_SAC = False  # set to True if you want to retrain SAC

    if TRAIN_SAC:
        sac_model, sac_cb = train(
            algo="sac",
            total_timesteps=200_000,
            seed=0,
            n_envs=8,
            antagonist_model=None,
        )
    else:
        from stable_baselines3 import SAC
        device = get_training_device()
        sac_model = SAC.load("foosball_sac_nenv8_model.zip", device=device)
        sac_cb = None  # not used later

    # ------------------- PHASE 2: Train TQC vs frozen SAC antagonist ------------------
    tqc_model, tqc_cb = train(
        algo="tqc",
        total_timesteps=200_000,
        seed=1,
        n_envs=8,
        antagonist_model=sac_model,
    )

    print("[DONE] SAC and TQC training complete.")

