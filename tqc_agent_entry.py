# tqc_agent_entry.py
import os
import argparse
import numpy as np
import torch

import math
import json
from pathlib import Path
# torch.set_default_dtype(torch.float32)

from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TransformObservation

from ai_agents.common.train.impl.generic_agent_manager import GenericAgentManager
from ai_agents.common.train.impl.single_player_training_engine import SinglePlayerTrainingEngine
from ai_agents.common.train.impl.tqc_agent import TQCFoosballAgent
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from ai_agents.common.train.impl.performance_utils import setup_performance_optimizations


# ---- env factory (identical wrapping to SAC path) ----
def tqc_foosball_env_factory(_=None):
    env = FoosballEnv(antagonist_model=None)
    env = Monitor(env)
    return env


def evaluate_tqc_model(
    model_dir: str = "./models",
    log_dir: str = "./logs",
    agent_id: int = 0,
    n_episodes: int = 20,
    max_steps: int = 1000,
):
    """
    Evaluate the best TQC model and log summary metrics:

      - 'average_reward'
      - 'average_last_distance_to_goal'
      - 'average_ball_speed'

    The summary is printed and also written to
      <log_dir>/tqc_eval_summary.json
    so you can drop it directly into your report.
    """
    os.makedirs(log_dir, exist_ok=True)

    # fresh eval env (Monitor-wrapped)
    env = tqc_foosball_env_factory()

    # 🔎 unwrap Monitor (and any other wrappers) to get the underlying FoosballEnv
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # bind a TQCFoosballAgent to this env and load best model if present
    agent = TQCFoosballAgent(
        id=agent_id,
        env=env,
        log_dir=log_dir,
        model_dir=model_dir,
    )
    agent.initialize_agent()  # tries to load best_model; otherwise new model

    episode_rewards = []
    episode_last_dists = []
    episode_mean_speeds = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        speeds = []
        last_dist = None

        for step in range(max_steps):
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            ep_reward += float(reward)

            # Use env helpers from the underlying FoosballEnv
            ball_pos, ball_vel = base_env._get_ball_obs()
            bx, by = ball_pos
            vx, vy = ball_vel

            speed = math.sqrt(vx * vx + vy * vy)
            speeds.append(speed)

            # "Last distance to goal" = distance from current ball position
            # to the (0, TABLE_MAX_Y_DIM) target
            last_dist = base_env.euclidean_goal_distance(bx, by)

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_last_dists.append(last_dist if last_dist is not None else float("nan"))
        episode_mean_speeds.append(
            float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        )

    avg_reward = float(np.mean(episode_rewards))
    avg_last_dist = float(np.nanmean(episode_last_dists))
    avg_speed = float(np.mean(episode_mean_speeds))

    summary = {
        "n_episodes": n_episodes,
        "average_reward": avg_reward,
        "average_last_distance_to_goal": avg_last_dist,
        "average_ball_speed": avg_speed,
    }

    print(
        f"[TQC EVAL] over {n_episodes} episodes:\n"
        f"  Average Reward                = {avg_reward:.3f}\n"
        f"  Avg. Last Distance to Goal    = {avg_last_dist:.3f}\n"
        f"  Average Ball Speed (velocity) = {avg_speed:.5f}"
    )

    out_path = Path(log_dir) / "tqc_eval_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[TQC EVAL] wrote summary metrics to {out_path}")

    return summary


if __name__ == "__main__":
    # Performance optimizations for RTX 5090 and Ryzen 9 9950X3D
    device = setup_performance_optimizations(num_threads=32, num_interop_threads=8)

    # Recommended for mac (kept for compatibility, but RTX 5090 uses CUDA)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    parser = argparse.ArgumentParser(description="Train or test TQC model.")
    parser.add_argument("-t", "--test", help="Test mode", action="store_true")
    args = parser.parse_args()

    model_dir = "./models"
    log_dir = "./logs"
    total_epochs = 15
    epoch_timesteps = int(100000)

    # Same orchestration as SAC, just swap the Agent class
    agent_manager = GenericAgentManager(1, tqc_foosball_env_factory, TQCFoosballAgent)
    agent_manager.initialize_training_agents()
    agent_manager.initialize_frozen_best_models()

    engine = SinglePlayerTrainingEngine(
        agent_manager=agent_manager,
        environment_generator=tqc_foosball_env_factory,
    )

    if not args.test:
        engine.train(
            total_epochs=total_epochs,
            epoch_timesteps=epoch_timesteps,
            cycle_timesteps=10000,
        )

    # Run whatever built-in test loop you already rely on
    engine.test()

    # And then run explicit metric-based evaluation (parallel to SAC)
    evaluate_tqc_model(
        model_dir=model_dir,
        log_dir=log_dir,
        agent_id=0,        # first (and only) agent in GenericAgentManager
        n_episodes=20,
        max_steps=1000,
    )
