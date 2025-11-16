from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import numpy as np

env = FoosballEnv(antagonist_model=None, play_until_goal=False,
                  verbose_mode=True)
obs, info = env.reset()

# Put ball at center & give it a velocity
env._reset_ball_to_center()
env._kick_ball_x(vx=2.0)

ball_pos, ball_vel = env._get_ball_obs()
print("[DEBUG] after kick, pos=", ball_pos, "vel=", ball_vel)

for t in range(50):
    # no rod movement at all
    action = np.zeros(env.protagonist_action_size, dtype=np.float64)
    obs, reward, terminated, truncated, info = env.step(action)
    ball_pos, ball_vel = env._get_ball_obs()
    print(
        f"t={t:03d} pos=({ball_pos[0]:.3f}, {ball_pos[1]:.3f}) "
        f"vel=({ball_vel[0]:.3f}, {ball_vel[1]:.3f})"
    )


