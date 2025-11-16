# ai_agents/common/train/impl/tqc_agent.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict
import numpy as np

from sb3_contrib import TQC  # pip install sb3-contrib
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from tqdm import tqdm

from ai_agents.common.train.interface.foosball_agent import FoosballAgent
from ai_agents.common.train.impl.performance_utils import get_device, compile_model_if_supported


class TqdmCallback(BaseCallback):
    """Progress bar for SB3."""
    def __init__(self, total_timesteps=None, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        total = self.locals.get("total_timesteps", self.total_timesteps)
        self.pbar = tqdm(total=total, desc="Training (TQC)", unit="step") if total else tqdm(desc="Training (TQC)", unit="step")

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

class FoosballTrainingMonitorCallback(BaseCallback):
    """
    Logs foosball-specific metrics:
      - mean episode reward (from ep_info_buffer)
      - mean max |ball_y| per episode (how far the ball travels down the table)
      - approximate goal rate (episodes where |max_y| > threshold)
    """
    def __init__(self, window: int = 50, log_every_steps: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.window = window
        self.log_every_steps = log_every_steps

        # per-env ephemeral state
        self.current_max_y = None
        # episode-level buffers
        self.episode_max_y = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs if hasattr(self.training_env, "num_envs") else 1
        self.current_max_y = np.zeros(n_envs, dtype=np.float32)

    def _on_step(self) -> bool:
        # infos: list of dicts, one per env
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if len(infos) == 0:
            return True

        n_envs = len(infos)
        # grow state if VecEnv size changed for some reason
        if self.current_max_y is None or len(self.current_max_y) != n_envs:
            self.current_max_y = np.zeros(n_envs, dtype=np.float32)

        for i in range(n_envs):
            info = infos[i] or {}
            ball_y = info.get("ball_y", None)
            if ball_y is not None:
                # track max |y| for this episode/env
                self.current_max_y[i] = max(self.current_max_y[i], abs(float(ball_y)))

            done = bool(dones[i]) if isinstance(dones, (list, np.ndarray)) else False
            if done:
                # end of episode for env i
                self.episode_max_y.append(self.current_max_y[i])
                self.current_max_y[i] = 0.0

        # Periodic logging
        if self.num_timesteps % self.log_every_steps == 0 and len(self.episode_max_y) > 0:
            window_max_y = self.episode_max_y[-self.window:]
            mean_abs_max_y = float(np.mean(window_max_y))

            # approximate goal rate: episodes where |max_y| is close to table end
            GOAL_Y_THRESHOLD = 60.0  # TABLE_MAX_Y_DIM ~65 → 60 is “almost goal”
            goal_flags = [1.0 if y > GOAL_Y_THRESHOLD else 0.0 for y in window_max_y]
            goal_rate = float(np.mean(goal_flags))

            # mean episodic reward from SB3's ep_info_buffer
            mean_ep_reward = None
            if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                rewards_window = [ep_info["r"] for ep_info in list(self.model.ep_info_buffer)[-self.window:]]
                mean_ep_reward = float(np.mean(rewards_window))

            # log to TensorBoard
            self.logger.record("foosball/mean_abs_max_ball_y", mean_abs_max_y)
            self.logger.record("foosball/goal_rate", goal_rate)
            if mean_ep_reward is not None:
                self.logger.record("foosball/mean_episode_reward_window", mean_ep_reward)

            if self.verbose > 0:
                msg = f"[FOOSBALL TRAIN] steps={self.num_timesteps} "
                if mean_ep_reward is not None:
                    msg += f"mean_ep_reward(last_{self.window})={mean_ep_reward:.1f} "
                msg += f"mean_max_|y|(last_{self.window})={mean_abs_max_y:.2f} "
                msg += f"goal_rate(last_{self.window})={goal_rate:.2%}"
                print(msg)

        return True

class TQCFoosballAgent(FoosballAgent):
    """
    Mirrors SACFoosballAgent but uses sb3-contrib TQC.
    Keeps *exact* SAC config knobs: device='mps', buffer_size=1_000_000, policy_kwargs passthrough, same EvalCallback.
    """
    def __init__(
        self,
        id: int,
        env=None,
        log_dir: str = "./logs",
        model_dir: str = "./models",
         policy_kwargs: Dict = dict(net_arch=[512, 512]),
    ):
        self.env = env
        self.model: TQC | None = None
        self.id = id
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.id_subdir = f"{model_dir}/{id}"
        self.policy_kwargs = policy_kwargs

        os.makedirs(self.id_subdir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_id(self):
        return self.id

    # ---------- SB3 save/load helpers (avoid .zip.zip) ----------
    def _best_model_basepath(self) -> Path:
        return Path(self.id_subdir) / "tqc" / "best_model" / "best_model"

    def save(self) -> None:
        base = self._best_model_basepath()
        base.parent.mkdir(parents=True, exist_ok=True)
        assert self.model is not None, "Model not initialized"
        self.model.save(str(base))  # SB3 appends .zip

    def load(self) -> None:
        base = self._best_model_basepath()
        device = get_device()
        self.model = TQC.load(str(base), device=str(device))
        # Compile model for improved performance if supported
        if self.model.policy is not None:
            try:
                self.model.policy = compile_model_if_supported(self.model.policy)
            except Exception:
                pass  # Skip compilation if it fails, use uncompiled model
        print(f"Agent {self.id} loaded model from {base}.zip")

    # ---------- Init / Learn / Predict ----------
    def initialize_agent(self) -> None:
        try:
            self.load()
        except Exception:
            print(f"Agent {self.id} could not load model. Initializing new TQC model.")
            device = get_device()
            self.model = TQC(
                "MlpPolicy",
                self.env,
                # keep SAC-aligned core knobs:
                buffer_size=300_000,
                device=str(device),
                policy_kwargs=self.policy_kwargs,
                verbose=0,
                tensorboard_log=self.log_dir,
            )
            # Compile model for improved performance if supported
            if self.model.policy is not None:
                try:
                    self.model.policy = compile_model_if_supported(self.model.policy)
                except Exception:
                    pass  # Skip compilation if it fails, use uncompiled model
            # NOTE: Stable-Baselines3 stores replay buffer on same device as model (CUDA).
            # There is no built-in parameter to store buffer on CPU while model is on GPU.
            # The main memory leak fix (preventing agent accumulation) should resolve GPU memory issues.
            # If memory is still tight, consider reducing buffer_size (e.g., 500_000 or 250_000).
        print("SAC/TQC device:", self.model.device if self.model else "N/A")  # sanity
        print(f"Agent {self.id} initialized (TQC).")

    def predict(self, observation, deterministic: bool = False):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def create_callback(self, env, total_timesteps=None, show_progress=True):
        eval_cb = EvalCallback(
            env,
            best_model_save_path=str(self._best_model_basepath().parent),
            log_path=self.log_dir,
            eval_freq=3000,
            n_eval_episodes=10,
            render=False,
            deterministic=True,
        )
        foosball_monitor = FoosballTrainingMonitorCallback(
            window=50,
            log_every_steps=10_000,
            verbose=1,
        )

        callbacks = [eval_cb, foosball_monitor]
        if show_progress and os.getenv("DISPLAY_PROGRESS", "1") != "0":
            callbacks.insert(0, TqdmCallback(total_timesteps=total_timesteps))

        if len(callbacks) == 1:
            return callbacks[0]
        return CallbackList(callbacks)


    def learn(self, total_timesteps: int) -> None:
        if self.model is None:
            # if someone calls learn() without initialize_agent()
            device = get_device()
            self.model = TQC(
                "MlpPolicy",
                self.env,
                buffer_size=1_000_000,
                batch_size=512,
                device=str(device),
                policy_kwargs=self.policy_kwargs,
                verbose=0,
                tensorboard_log=self.log_dir,
            )
            # Compile model for improved performance if supported
            if self.model.policy is not None:
                try:
                    self.model.policy = compile_model_if_supported(self.model.policy)
                except Exception:
                    pass  # Skip compilation if it fails, use uncompiled model
            # NOTE: Stable-Baselines3 stores replay buffer on same device as model (CUDA).
            # There is no built-in parameter to store buffer on CPU while model is on GPU.
            # The main memory leak fix (preventing agent accumulation) should resolve GPU memory issues.
            # If memory is still tight, consider reducing buffer_size (e.g., 500_000 or 250_000).
        callback = self.create_callback(self.env, total_timesteps=total_timesteps)
        tb_log_name = f"tqc_{self.id}"
        self.model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=tb_log_name)

    def change_env(self, env) -> None:
        """Swap the active Gym env (used by GenericAgentManager.set_agent_environment)."""
        self.env = env
        if self.model is not None:
            # SB3 provides set_env to rebind the environment safely.
            self.model.set_env(env)
