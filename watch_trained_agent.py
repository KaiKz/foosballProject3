# watch_selfplay_foosball.py
import time
from stable_baselines3 import SAC
from sb3_contrib import TQC

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


ALGO = "sac"  # or "tqc"

# Default model names (no .zip)
BLACK_MODEL_PATH = f"{ALGO}_foosball_round0_static"
YELLOW_MODEL_PATH = f"{ALGO}_foosball_round1_vs_round0"


def watch_selfplay(
    protagonist_path=YELLOW_MODEL_PATH,
    antagonist_path=BLACK_MODEL_PATH,
    algo=ALGO,
):
    if algo == "sac":
        yellow_model = SAC.load(protagonist_path, device="mps")
        black_model = SAC.load(antagonist_path, device="mps")
    else:
        yellow_model = TQC.load(protagonist_path, device="mps")
        black_model = TQC.load(antagonist_path, device="mps")

    env = FoosballEnv(
        antagonist_model=black_model,
        play_until_goal=True,
        verbose_mode=False,
        render_mode="human",
    )

    obs, info = env.reset()

    try:
        while True:
            action, _ = yellow_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)

            if terminated or truncated:
                print(f"[WATCH] Episode ended, reward={info.get('reward', 0.0):.3f}")
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    watch_selfplay()
