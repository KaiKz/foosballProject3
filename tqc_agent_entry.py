# tqc_agent_entry.py
import os
import argparse
import numpy as np
import torch

# torch.set_default_dtype(torch.float32)

from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TransformObservation

from ai_agents.common.train.impl.generic_agent_manager import GenericAgentManager
from ai_agents.common.train.impl.single_player_training_engine import SinglePlayerTrainingEngine
from ai_agents.common.train.impl.tqc_agent import TQCFoosballAgent
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from ai_agents.common.train.impl.performance_utils import setup_performance_optimizations

# ---- env factory (identical wrapping to SAC path) ----
def tqc_foosball_env_factory(_=None):
    env = FoosballEnv(antagonist_model=None)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # Performance optimizations for RTX 5090 and Ryzen 9 9950X3D
    device = setup_performance_optimizations(num_threads=32, num_interop_threads=8)
    
    # Recommended for mac (kept for compatibility, but RTX 5090 uses CUDA)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    parser = argparse.ArgumentParser(description="Train or test TQC model.")
    parser.add_argument("-t", "--test", help="Test mode", action="store_true")
    args = parser.parse_args()

    model_dir = "./models"
    log_dir = "./logs"
    total_epochs = 15
    epoch_timesteps = int(100000)

    # Same orchestration as SAC, just swap the Agent class
    agent_manager = GenericAgentManager(1, tqc_foosball_env_factory, TQCFoosballAgent)
    agent_manager.initialize_training_agents()
    agent_manager.initialize_frozen_best_models()

    engine = SinglePlayerTrainingEngine(
        agent_manager=agent_manager,
        environment_generator=tqc_foosball_env_factory,
    )

    if not args.test:
        engine.train(total_epochs=total_epochs, epoch_timesteps=epoch_timesteps, cycle_timesteps=10000)

    engine.test()
