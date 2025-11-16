# debug_random_rollout.py
import numpy as np
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

env = FoosballEnv(
    antagonist_model=None,
    verbose_mode=True,
    debug_free_ball=True,  # keep the extra logging + ball tweaks for now
)
obs, info = env.reset()

for t in range(50):
    # random actions in the allowed range
    action = np.random.uniform(
        low=env.action_space.low,
        high=env.action_space.high,
        size=env.protagonist_action_size,
    ).astype(np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"t={t:02d} ball_x={info['ball_x']:.3f} "
        f"ball_y={info['ball_y']:.3f} reward={info['reward']:.3f}"
    )

    if terminated or truncated:
        print("Episode ended at t=", t)
        break

