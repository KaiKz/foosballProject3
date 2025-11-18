# sac_agent_entry.py
import os
import math
import json
import argparse
from pathlib import Path

import numpy as np

from stable_baselines3.common.monitor import Monitor

from ai_agents.common.train.impl.generic_agent_manager import GenericAgentManager
from ai_agents.common.train.impl.single_player_training_engine import SinglePlayerTrainingEngine
from ai_agents.common.train.impl.sac_agent import SACFoosballAgent
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from ai_agents.common.train.impl.performance_utils import setup_performance_optimizations


# ---- env factory (identical style to TQC: single-agent, no antagonist) ----
def sac_foosball_env_factory(_=None):
    env = FoosballEnv(antagonist_model=None)
    env = Monitor(env)
    return env


def evaluate_sac_model(
    model_dir: str = "./models",
    log_dir: str = "./logs",
    agent_id: int = 0,
    n_episodes: int = 20,
    max_steps: int = 1000,
):
    """
    Evaluate the best SAC model and log summary metrics:

      - 'average_reward'
      - 'average_last_distance_to_goal'
      - 'average_ball_speed'

    The summary is printed and also written to
      <log_dir>/sac_eval_summary.json
    so you can drop it directly into your report.
    """
    os.makedirs(log_dir, exist_ok=True)

    # fresh eval env (no VecEnv; we use plain Gym API)
    env = sac_foosball_env_factory()
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # bind a SACFoosballAgent to this env and load best model if present
    agent = SACFoosballAgent(
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

            # Use env helpers to get ball pos/vel
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
        f"[SAC EVAL] over {n_episodes} episodes:\n"
        f"  Average Reward                = {avg_reward:.3f}\n"
        f"  Avg. Last Distance to Goal    = {avg_last_dist:.3f}\n"
        f"  Average Ball Speed (velocity) = {avg_speed:.5f}"
    )

    out_path = Path(log_dir) / "sac_eval_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAC EVAL] wrote summary metrics to {out_path}")

    return summary


if __name__ == "__main__":
    # Performance optimizations for RTX 5090 and Ryzen 9 9950X3D
    device = setup_performance_optimizations(num_threads=32, num_interop_threads=8)

    # Keep same env variable setup style as TQC
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    parser = argparse.ArgumentParser(description="Train or test SAC model.")
    parser.add_argument("-t", "--test", help="Test mode", action="store_true")
    args = parser.parse_args()

    model_dir = "./models"
    log_dir = "./logs"
    total_epochs = 15
    epoch_timesteps = int(100_000)

    # Same orchestration as TQC, just SACFoosballAgent instead
    agent_manager = GenericAgentManager(1, sac_foosball_env_factory, SACFoosballAgent)
    agent_manager.initialize_training_agents()
    agent_manager.initialize_frozen_best_models()

    engine = SinglePlayerTrainingEngine(
        agent_manager=agent_manager,
        environment_generator=sac_foosball_env_factory,
    )

    if not args.test:
        engine.train(
            total_epochs=total_epochs,
            epoch_timesteps=epoch_timesteps,
            cycle_timesteps=10_000,
        )

    # Same style as TQC: run your existing test loop
    engine.test()

    # Then run explicit metric-based evaluation
    evaluate_sac_model(
        model_dir=model_dir,
        log_dir=log_dir,
        agent_id=0,        # first (and only) agent in GenericAgentManager
        n_episodes=20,
        max_steps=1000,
    )
