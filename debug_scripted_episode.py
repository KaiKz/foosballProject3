#!/usr/bin/env python

import math
import time

import numpy as np

# 🔴 IMPORTANT: adjust this import to match your project layout.
# If your env is in ai_agents/v2/gym/foosball_env.py:
try:
    from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
except ImportError:
    # fallback if you have it as foosball_env.py in the root
    from foosball_env import FoosballEnv


def run_debug_episode(num_episodes: int = 1, sleep_per_step: float = 0.02):
    """
    Run a few episodes with a simple scripted protagonist policy:

    - Only the attack rod (indices 6 and 7) moves.
    - It oscillates sinusoidally to try to create ball contact.
    - We render with the MuJoCo viewer ("human" mode).
    """

    # play_until_goal=False so we still allow stagnation/time-based termination
    env = FoosballEnv(
        play_until_goal=False,
        verbose_mode=True,
        render_mode="human",
        kick_drill_mode=True,
    )

    try:
        for ep in range(num_episodes):
            print(f"\n=== DEBUG EPISODE {ep} ===")
            obs, info = env.reset(seed=ep)

            done = False
            t = 0

            while not done:
                # 8-dim protagonist action:
                # [goal_lin, goal_rot, def_lin, def_rot,
                #  mid_lin, mid_rot, attack_lin, attack_rot]
                a = np.zeros(env.protagonist_action_size, dtype=np.float32)

                # simple oscillation to move attack rod back and forth
                phase = math.sin(0.1 * t)
                # a[6] = 0.3 * phase   # attack linear
                a[7] = -8 * phase   # attack rotation

                obs, reward, done, truncated, info = env.step(a)

                # render to the MuJoCo GUI window
                env.render()

                # pull ball + termination info
                ball_x = info.get("ball_x", float("nan"))
                ball_y = info.get("ball_y", float("nan"))

                # only print detailed info every 10 steps or when the episode ends
                if t % 10 == 0 or done:
                    # termination reasons if you added _last_termination_info in the env
                    term_info = {
                        k: info.get(k, None)
                        for k in [
                            "winning_goal",
                            "losing_goal",
                            "ball_stagnant",
                            "out_of_play",
                            "over_max_time",
                        ]
                        if k in info
                    }

                    print(
                        f"t={t:4d} | "
                        f"ball=({ball_x:6.2f}, {ball_y:6.2f}) | "
                        f"r={reward:7.3f} | "
                        f"info_extra={term_info}"
                    )

                t += 1
                time.sleep(sleep_per_step)

            print(f"Episode {ep} finished at t={t} steps")

    finally:
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    # You can tweak num_episodes / sleep here
    run_debug_episode(num_episodes=1, sleep_per_step=0.02)
