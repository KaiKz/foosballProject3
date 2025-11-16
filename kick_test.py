import numpy as np
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

env = FoosballEnv(antagonist_model=None, verbose_mode=True)
obs, _ = env.reset()

# 1) Put ball near a specific rod if needed
# for example, you can tweak reset to place ball at some known (x, y) near yellow mid rod

for t in range(50):
    action = np.zeros(env.protagonist_action_size, dtype=np.float64)
    # Max push on first linear joint (or one you know is near the ball)
    action[0] = 20.0  # or whatever is your max for that rod

    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"t={t:03d} ball_x={info['ball_x']:.3f} "
        f"ball_y={info['ball_y']:.3f} reward={reward:.3f}"
    )
    if terminated or truncated:
        break
