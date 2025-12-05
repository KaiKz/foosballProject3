# sac_watch_entry_v2.py
import argparse
import time

import numpy as np
import torch

from ai_agents.common.train.impl.performance_utils import setup_performance_optimizations
from ai_agents.common.train.impl.sac_agent import SACFoosballAgent
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def make_watch_env():
    """
    Create a FoosballEnv configured for *interactive watching*.
    The key part is render_mode="human".
    """
    env = FoosballEnv(
        antagonist_model=None,
        play_until_goal=False,
        verbose_mode=False,      # set True if you want spammy debug prints
        debug_free_ball=False,
        render_mode="human",     # <-- CRITICAL for viewer popup
    )
    return env


def watch(model_id: int, episodes: int, sleep_per_step: float) -> None:
    # Use the same performance setup as training (mainly to pick device)
    setup_performance_optimizations(num_threads=8, num_interop_threads=4)

    # Create env with human viewer
    env = make_watch_env()

    # Create the agent and load the trained SAC model
    agent = SACFoosballAgent(id=model_id, env=env)
    agent.initialize_agent()   # will call .load() internally if checkpoint exists

    print(f"[WATCH] Loaded model for agent {model_id}. Starting episodes...")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_idx = 0

        print(f"\n[WATCH] Episode {ep + 1}/{episodes} start")

        while not done:
            # Deterministic policy for watching
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            done = bool(terminated or truncated)

            # *** THIS is what actually opens and updates the Mujoco viewer ***
            env.render()

            # Optional: slow down for human eyes
            if sleep_per_step > 0.0:
                time.sleep(sleep_per_step)

            step_idx += 1

        print(f"[WATCH] Episode {ep + 1} finished in {step_idx} steps, "
              f"total_reward={ep_reward:.2f}")

    env.close()
    print("[WATCH] All episodes done. Viewer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int, default=0, help="Agent/model id")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch")
    parser.add_argument("--sleep", type=float, default=0.02, help="Sleep per step (seconds)")
    args = parser.parse_args()

    # Small safety: ensure default dtype is float32 like in training
    torch.set_default_dtype(torch.float32)
    np.set_printoptions(precision=4, suppress=True)

    watch(
        model_id=args.model_id,
        episodes=args.episodes,
        sleep_per_step=args.sleep,
    )
