# random_rollout_sanity.py
import numpy as np
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

env = FoosballEnv(
    antagonist_model=None,
    play_until_goal=False,
    verbose_mode=False,   # or True if you want spam
    debug_free_ball=False # real training settings
)

num_episodes = 5
max_steps    = 500

for ep in range(num_episodes):
    obs, info = env.reset()
    ep_rewards = 0.0
    ball_path = []

    for t in range(max_steps):
        action = env.action_space.sample()  # random protagonist
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rewards += reward
        ball_path.append((info["ball_x"], info["ball_y"]))

        if terminated or truncated:
            break

    ball_xs = [p[0] for p in ball_path]
    ball_ys = [p[1] for p in ball_path]

    print(f"Episode {ep}: steps={len(ball_path)}, return={ep_rewards:.2f}, "
          f"ball_x_range=({min(ball_xs):.3f},{max(ball_xs):.3f}), "
          f"ball_y_range=({min(ball_ys):.3f},{max(ball_ys):.3f})")
