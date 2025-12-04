from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import numpy as np
import time

env = FoosballEnv(antagonist_model=None, verbose_mode=True, render_mode="human")

obs, info = env.reset()

for t in range(20000):
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    # a[6] = 1.0   # slide attack rod
    # a[7] =  1.0    # rotate attack rod

    obs, r, terminated, truncated, info = env.step(a)
    env.render()

    if terminated or truncated:
        break

print("Simulation finished – keeping viewer open. Close the window or Ctrl+C in the terminal to quit.")

# Keep window open
t = 0
while True:
    # protagonist (yellow) actions
    a = np.zeros(env.protagonist_action_size, dtype=np.float32)

    # --- YELLOW ATTACK ROD TIMING ---
    # e.g. steps 20–40: yellow rod rotates to kick the ball
    if 20 <= t < 40:
        a[7] = -1.0    # y_attack_rotation (scale happens inside env)
        a[7] = 1.0 
    # --- BLACK ATTACK ROD TIMING ---
    # e.g. steps 80–100: black rod rotates to kick the ball back
    if 80 <= t < 100:
        env.data.ctrl[13] = -2.5   # b_attack_rotation (same scale as yellow rot_range)
    else:
        env.data.ctrl[13] = 0.0   # stop black rod when not “kicking”

    obs, reward, terminated, truncated, info = env.step(a)
    env.render()
    time.sleep(0.02)
    t += 1

    if terminated or truncated:
        # reset timing + state, and give yourself a moment to see reset
        obs, info = env.reset()
        env.render()
        time.sleep(2.0)
        t = 0
