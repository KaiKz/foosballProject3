from ai_agents.common.train.impl.sac_agent import SACFoosballAgent
from ai_agents.common.train.interface.agent_manager import AgentManager
import torch
import gc

class GenericAgentManager(AgentManager):
    def __init__(self, num_agents: int, environment_generator, agent_class):
        self.num_agents = num_agents
        self.environment_generator = environment_generator
        self.initial_env = environment_generator()
        self.agent_class = agent_class
        self.training_agents = []
        self.frozen_best_models = []

    def initialize_training_agents(self):
        for i in range(self.num_agents):
            agent = self.agent_class(id=i, env=self.initial_env)
            agent.initialize_agent()
            self.training_agents.append(agent)

    def save_training_agents(self):
        for agent in self.training_agents:
            agent.save()

    def initialize_frozen_best_models(self):
        # Clear old frozen models to prevent memory leak
        # This is the CRITICAL FIX: Without this, agents accumulate every epoch causing GPU OOM
        for old_agent in self.frozen_best_models:
            if old_agent.model is not None:
                # Try to move model components to CPU to free GPU memory
                try:
                    # Move policy networks to CPU
                    if hasattr(old_agent.model, 'policy') and old_agent.model.policy is not None:
                        old_agent.model.policy = old_agent.model.policy.cpu()
                    # Move target networks to CPU if they exist
                    if hasattr(old_agent.model, 'policy_target') and old_agent.model.policy_target is not None:
                        old_agent.model.policy_target = old_agent.model.policy_target.cpu()
                    # Clear replay buffer reference (it's on GPU, just clear the reference)
                    if hasattr(old_agent.model, 'replay_buffer') and old_agent.model.replay_buffer is not None:
                        old_agent.model.replay_buffer = None
                    # Clear optimizer if it exists
                    if hasattr(old_agent.model, 'actor_optimizer') and old_agent.model.actor_optimizer is not None:
                        old_agent.model.actor_optimizer = None
                    if hasattr(old_agent.model, 'critic_optimizer') and old_agent.model.critic_optimizer is not None:
                        old_agent.model.critic_optimizer = None
                except Exception:
                    pass  # Ignore errors during cleanup - just ensure model is cleared
                # Clear model reference
                old_agent.model = None
            # Clear environment reference
            old_agent.env = None
            del old_agent
        
        # Clear the list
        self.frozen_best_models.clear()
        
        # Force garbage collection and clear CUDA cache to free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create new frozen models
        for i in range(self.num_agents):
            agent = self.agent_class(id=i, env=self.initial_env)
            agent.initialize_agent()
            self.frozen_best_models.append(agent)

    def get_training_agents(self):
        return self.training_agents

    def get_frozen_best_models(self):
        return self.frozen_best_models

    def set_agent_environment(self, id, env):
        self.training_agents[id].change_env(env)

    def set_training_agent(self, agent):
        self.training_agents.append(agent)

