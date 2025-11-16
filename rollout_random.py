import numpy as np
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

def main():
    env = FoosballEnv(verbose_mode=True)
    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)

    steps = 500
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if t < 20:
            print(
                f"t={t:03d} "
                f"ball_x={info.get('ball_x', float('nan')):.3f} "
                f"ball_y={info.get('ball_y', float('nan')):.3f} "
                f"reward={reward:.3f} "
                f"terminated={terminated} truncated={truncated}"
            )

        if terminated or truncated:
            print(f"Episode ended at t={t}, resetting...")
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
