from ai_agents.common.train.interface.agent_manager import AgentManager
from typing import List
from ai_agents.common.train.interface.training_engine import TrainingEngine
from tqdm import tqdm

class ProtagonistAntagonistTrainingEngine(TrainingEngine):
    def __init__(
            self,
            agent_manager: AgentManager,
            environment_generator

    ):
        self.agent_manager = agent_manager
        self.current_epoch = 0
        self.best_models: List[str] = []
        self.num_agents_training = len(self.agent_manager.get_training_agents())
        self.environment_generator = environment_generator

    def train(self, total_epochs: int, epoch_timesteps: int, cycle_timesteps: int):
        # Initialize frozen models once at the start (they will be updated by EvalCallback)
        # Only reload if we need the latest best model, but not every epoch to save memory
        self.agent_manager.initialize_frozen_best_models()
        
        with tqdm(total=total_epochs, desc="Overall Training", unit="epoch") as pbar_epoch:
            for epoch in range(total_epochs):
                pbar_epoch.set_description(f"Epoch {epoch + 1}/{total_epochs}")
                protagonist_agents = self.agent_manager.get_training_agents()
                # Only reload frozen models every few epochs to reduce memory churn
                # EvalCallback saves best model automatically, so we don't need to reload every epoch
                if epoch == 0 or epoch % 5 == 0:  # Reload every 5 epochs instead of every epoch
                    self.agent_manager.initialize_frozen_best_models()
                antagonist_agents = self.agent_manager.get_frozen_best_models()

                envs_to_close = []  # Track environments to close
                for cycle in range(self.num_agents_training):
                    ## Train the first protagonist agent for now.
                    protagonist_agent = protagonist_agents[0]
                    antagonist_agent = antagonist_agents[cycle]

                    env = self.environment_generator(antagonist_agent)
                    envs_to_close.append(env)
                    protagonist_agent.change_env(env)
                    protagonist_agent.learn(epoch_timesteps)
                
                # Close all environments created this epoch
                for env in envs_to_close:
                    try:
                        env.close()
                    except Exception:
                        pass  # Ignore errors if close() not implemented or fails

                self.current_epoch += 1
                pbar_epoch.update(1)


    def test(self, num_episodes: int = 100):
        protagonist = self.agent_manager.get_frozen_best_models()[0]
        env = self.environment_generator(protagonist)

        try:
            for episode in tqdm(range(num_episodes), desc="Testing", unit="episode"):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = protagonist.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    env.render()
                    done = terminated or truncated
        finally:
            # Ensure environment is closed even if test is interrupted
            try:
                env.close()
            except Exception:
                pass