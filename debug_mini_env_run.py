import numpy as np
from foosball_mini_env import FoosballMiniEnv


def run_policy(env, policy_fn, episodes=1, render=False, label="policy"):
    all_returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_return = 0.0
        step = 0

        print(f"\n--- {label} | Episode {ep} ---")
        print("step\tball_x\tball_y\trod_x\treward\taction")

        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            bx = info["ball_x"]
            by = info["ball_y"]
            rx = info["rod_x"]

            print(
                f"{step}\t{bx:.3f}\t{by:.3f}\t{rx:.3f}\t{reward:.4f}\t{float(action[0]):.3f}"
            )

            ep_return += reward
            step += 1

            if render:
                env.render()

        print(f"{label} | Episode {ep} return: {ep_return:.4f}")
        all_returns.append(ep_return)

    print(f"\n[{label}] mean return over {episodes} episodes: {np.mean(all_returns):.4f}")
    return all_returns


def policy_random(obs):
    return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)


def policy_push_right(obs):
    return np.array([1.0], dtype=np.float32)


def policy_push_left(obs):
    return np.array([-1.0], dtype=np.float32)


if __name__ == "__main__":
    env = FoosballMiniEnv(render_mode=None, frame_skip=5, max_steps=200)

    print("\n=== Running RANDOM policy ===")
    run_policy(env, policy_random, episodes=3, label="random")

    print("\n=== Running PUSH_RIGHT policy ===")
    run_policy(env, policy_push_right, episodes=1, label="push_right")

    print("\n=== Running PUSH_LEFT policy ===")
    run_policy(env, policy_push_left, episodes=1, label="push_left")

    env.close()
