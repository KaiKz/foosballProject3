# ai_agents/common/train/impl/sac_agent.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from tqdm import tqdm

from ai_agents.common.train.interface.foosball_agent import FoosballAgent
from ai_agents.common.train.impl.performance_utils import get_device, compile_model_if_supported


class TqdmCallback(BaseCallback):
    """Progress bar for SB3 (SAC)."""
    def __init__(self, total_timesteps=None, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        total = self.locals.get("total_timesteps", self.total_timesteps)
        self.pbar = (
            tqdm(total=total, desc="Training (SAC)", unit="step")
            if total
            else tqdm(desc="Training (SAC)", unit="step")
        )

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class SACFoosballAgent(FoosballAgent):
    """
    SAC version that mirrors TQCFoosballAgent in style and config.

    - Same device handling (get_device + optional compile).
    - Same buffer_size / batch_size settings as TQC.
    - Same save/load layout, just under 'sac/'.
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
        self.model: SAC | None = None
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
        """
        Base path *without* the .zip extension.

        EvalCallback will save:
          <id_subdir>/sac/best_model/best_model.zip

        We keep that same convention for manual save/load.
        """
        return Path(self.id_subdir) / "sac" / "best_model" / "best_model"

    def save(self) -> None:
        base = self._best_model_basepath()
        base.parent.mkdir(parents=True, exist_ok=True)
        assert self.model is not None, "Model not initialized"
        # SB3 appends ".zip"
        self.model.save(str(base))

    def load(self) -> None:
        base = self._best_model_basepath()
        device = get_device()
        self.model = SAC.load(str(base), device=str(device))
        # Compile model for improved performance if supported
        if self.model.policy is not None:
            try:
                self.model.policy = compile_model_if_supported(self.model.policy)
            except Exception:
                # If compilation fails, just keep the original policy
                pass
        print(f"Agent {self.id} loaded model from {base}.zip")

    # ---------- Init / Learn / Predict ----------
    def initialize_agent(self) -> None:
        try:
            self.load()
        except Exception:
            print(f"Agent {self.id} could not load model. Initializing new SAC model.")
            device = get_device()
            self.model = SAC(
                "MlpPolicy",
                self.env,
                # same core knobs as TQC initialize_agent
                buffer_size=300_000,
                device=str(device),
                policy_kwargs=self.policy_kwargs,
                verbose=0,
                tensorboard_log=self.log_dir,
            )
            if self.model.policy is not None:
                try:
                    self.model.policy = compile_model_if_supported(self.model.policy)
                except Exception:
                    pass  # Skip compilation if it fails
        print("SAC device:", self.model.device if self.model else "N/A")
        print(f"Agent {self.id} initialized (SAC).")

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

        callbacks = [eval_cb]
        if show_progress and os.getenv("DISPLAY_PROGRESS", "1") != "0":
            callbacks.insert(0, TqdmCallback(total_timesteps=total_timesteps))

        if len(callbacks) == 1:
            return callbacks[0]
        return CallbackList(callbacks)

    def learn(self, total_timesteps: int) -> None:
        if self.model is None:
            # If someone calls learn() without initialize_agent()
            device = get_device()
            self.model = SAC(
                "MlpPolicy",
                self.env,
                buffer_size=1_000_000,
                batch_size=512,
                device=str(device),
                policy_kwargs=self.policy_kwargs,
                verbose=0,
                tensorboard_log=self.log_dir,
            )
            if self.model.policy is not None:
                try:
                    self.model.policy = compile_model_if_supported(self.model.policy)
                except Exception:
                    pass
        callback = self.create_callback(self.env, total_timesteps=total_timesteps)
        tb_log_name = f"sac_{self.id}"
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
        )

    def change_env(self, env) -> None:
        """Swap the active Gym env (used by GenericAgentManager.set_agent_environment)."""
        self.env = env
        if self.model is not None:
            # SB3 provides set_env to rebind the environment safely.
            self.model.set_env(env)
