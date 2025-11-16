import numpy as np
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

def main():
    env = FoosballEnv(antagonist_model=None)
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs = reset_out
        info = {}

    print("Initial reset info:", info)
    obs, info = env.reset()
    for t in range(1, 501):
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        ball_x = info.get("ball_x", float("nan"))
        ball_y = info.get("ball_y", float("nan"))
        print(f"t={t:03d}  reward={reward:.3f}  ball=({ball_x:.6f}, {ball_y:.6f})")

    # for t in range(1, 201):  # 200 steps
    #     # 1) RANDOM ACTIONS (pure env test)
    #     action = env.action_space.sample()

    #     step_out = env.step(action)
    #     if len(step_out) == 5:
    #         obs, reward, terminated, truncated, info = step_out
    #         done = terminated or truncated
    #     else:
    #         obs, reward, done, info = step_out

    #     ball_x = info.get("ball_x", float("nan"))
    #     ball_y = info.get("ball_y", float("nan"))

    #     print(
    #         f"t={t:03d}  "
    #         f"reward={reward:.3f}  "
    #         f"ball=({ball_x:.3f}, {ball_y:.3f})"
    #     )

        # if done:
        #     print("=== episode ended (terminated or truncated) ===")
        #     reset_out = env.reset()
        #     if isinstance(reset_out, tuple) and len(reset_out) == 2:
        #         obs, info = reset_out
        #     else:
        #         obs = reset_out

    env.close()

if __name__ == "__main__":
    main()
