# train_selfplay_foosball.py
import os
import time

from stable_baselines3 import SAC
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


# ---------------- CONFIG ----------------
ALGO = "sac"  # or "tqc"

# How long each round trains
TIMESTEPS_ROUND0 = 10_000   # protagonist vs static black
TIMESTEPS_ROUND1 = 10_000   # protagonist vs frozen round0

# Model name prefix (no .zip)
MODEL_PREFIX = f"{ALGO}_foosball"

ROUND0_MODEL = f"{MODEL_PREFIX}_round0_static"
ROUND1_MODEL = f"{MODEL_PREFIX}_round1_vs_round0"

# After training, auto-launch viewer?
RUN_WATCH_AFTER_TRAIN = True


def make_env(antagonist_model=None, render_mode=None, verbose_mode=False):
    """
    Build a FoosballEnv wrapped for SB3.
    - antagonist_model: SB3 policy for black (or None for static black)
    """
    def _thunk():
        env = FoosballEnv(
            antagonist_model=antagonist_model,
            play_until_goal=True,
            verbose_mode=verbose_mode,
            render_mode=render_mode,
        )
        return Monitor(env)
    return _thunk


def make_sb3_model(algo, env, tensorboard_log):
    """
    Create either SAC or TQC with sane defaults.
    """
    if algo == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            buffer_size=200_000,
            learning_rate=3e-4,
            batch_size=512,
            gamma=0.99,
            tau=0.02,
            train_freq=1,
            gradient_steps=1,
            device="mps",
            tensorboard_log=tensorboard_log,
        )
    else:
        model = TQC(
            "MlpPolicy",
            env,
            verbose=1,
            buffer_size=200_000,
            learning_rate=3e-4,
            batch_size=512,
            gamma=0.99,
            tau=0.02,
            top_quantiles_to_drop_per_net=2,
            n_critics=5,
            device="mps",
            tensorboard_log=tensorboard_log,
        )
    return model


def train_round(round_name, total_timesteps, antagonist_model=None):
    """
    Train a new protagonist policy against a given antagonist_model.
    Returns the trained SB3 model and the save path (without .zip).
    """
    print(f"\n===============================")
    print(f"  TRAINING ROUND: {round_name}")
    print(f"  Opponent: {'frozen policy' if antagonist_model is not None else 'static'}")
    print(f"===============================\n")

    vec_env = DummyVecEnv([make_env(antagonist_model=antagonist_model, render_mode=None)])

    tb_log = f"./tb_foosball_{ALGO}_{round_name}/"
    model = make_sb3_model(ALGO, vec_env, tb_log)

    model.learn(total_timesteps=total_timesteps)

    save_path = f"{MODEL_PREFIX}_{round_name}"
    model.save(save_path)
    print(f"[SAVE] Trained model saved to: {save_path}.zip")

    vec_env.close()
    return model, save_path


def load_frozen_model(path_no_zip):
    """
    Load a frozen opponent model from disk, on mps device.
    """
    if ALGO == "sac":
        return SAC.load(path_no_zip, device="mps")
    else:
        return TQC.load(path_no_zip, device="mps")


def watch_selfplay(protagonist_path, antagonist_path):
    """
    Quick viewer: protagonist (yellow) vs antagonist (black).
    You can run the same logic from watch_selfplay_foosball.py.
    """
    print("\n[WATCH] Loading models for self-play...")

    if ALGO == "sac":
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
    # ---------- ROUND 0: protagonist vs static black ----------
    round0_model, round0_path = train_round(
        round_name="round0_static",
        total_timesteps=TIMESTEPS_ROUND0,
        antagonist_model=None,  # black rods fixed
    )

    # Reload the round0 model as a frozen opponent
    frozen_black = load_frozen_model(round0_path)

    # ---------- ROUND 1: protagonist vs frozen round0 (black) ----------
    round1_model, round1_path = train_round(
        round_name="round1_vs_round0",
        total_timesteps=TIMESTEPS_ROUND1,
        antagonist_model=frozen_black,
    )

    print("\n[TRAINING DONE]")
    print(f"  Black (opponent): {round0_path}.zip")
    print(f"  Yellow (latest protagonist): {round1_path}.zip")

    if RUN_WATCH_AFTER_TRAIN:
        watch_selfplay(
            protagonist_path=round1_path,
            antagonist_path=round0_path,
        )
