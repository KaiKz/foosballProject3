# train_sac_foosball.py
import numpy as np
from stable_baselines3 import SAC
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

# ---------------- CONFIG ----------------
ALGO = "sac"         # or "tqc"
USE_FROZEN_BLACK = True   # <- set to True if you want black controlled by a pre-trained policy
BLACK_MODEL_PATH = "/Users/kaikaizhang/foosballProject3/sac_foosball_yellow"  # path to black policy if USE_FROZEN_BLACK


# ---------- (OPTIONAL) LOAD BLACK POLICY ----------
black_policy_model = None
if USE_FROZEN_BLACK:
    if ALGO == "sac":
        black_policy_model = SAC.load(BLACK_MODEL_PATH)
    else:
        black_policy_model = TQC.load(BLACK_MODEL_PATH)


# ---------- ENV FACTORY (for SB3) ----------
def make_train_env():
    def _thunk():
        # No viewer + no verbose prints for speed
        env = FoosballEnv(
            antagonist_model=black_policy_model,  # None or frozen opponent
            verbose_mode=False,
            render_mode=None,   # <- headless during training
        )
        return Monitor(env)
    return _thunk


if __name__ == "__main__":
    # Vectorized wrapper (SB3 expects VecEnv)
    train_env = DummyVecEnv([make_train_env()])

    # ---------- CHOOSE ALGO: SAC or TQC ----------
    if ALGO == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            buffer_size=200_000,
            learning_rate=3e-4,
            batch_size=512,
            gamma=0.99,
            tau=0.02,
            train_freq=1,
            device='mps',
            gradient_steps=1,
            tensorboard_log="./tb_foosball_sac/",
        )
        save_name = "sac_foosball_black"
    else:
        model = TQC(
            "MlpPolicy",
            train_env,
            verbose=1,
            buffer_size=200_000,
            learning_rate=3e-4,
            batch_size=512,
            gamma=0.99,
            tau=0.02,
            top_quantiles_to_drop_per_net=2,
            n_critics=5,
            device='mps',
            tensorboard_log="./tb_foosball_tqc/",
        )
        save_name = "tqc_foosball_black"

    # ---------- TRAIN ----------
    model.learn(total_timesteps=10_000)   # tweak this
    model.save(save_name)

    train_env.close()
